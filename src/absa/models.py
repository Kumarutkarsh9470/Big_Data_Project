"""Models compared in the project.

GloVe-based (BiLSTM encoder, embeddings frozen):
  lstm   : BiLSTM + aspect-aware attention, no graph (ablation: is the graph useful?)
  gcn    : ASGCN-style (Zhang et al., 2019) - BiLSTM -> L x GCN over the dependency
           graph (position weighted) -> aspect mask -> attention readout
  gat    : same, GCN layers replaced by multi-head graph attention restricted to
           dependency neighbours (Velickovic et al., 2018)
  rgat   : relation-aware GAT - dependency relation labels enter the attention
           logits and messages (in the spirit of Wang et al., 2020 R-GAT)

Transformer-based (RoBERTa-base, fine-tuned end-to-end):
  roberta     : sentence-pair classifier [<s> sentence </s></s> aspect </s>]
  roberta-gcn : RoBERTa word states -> GCN over dependency graph -> aspect pooling
  roberta-gat : same with relation-aware GAT
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ graph layers
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, h, adj):
        # adj: (B, N, N) with self loops; row-normalised mean aggregation (ASGCN)
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        return F.relu(torch.bmm(adj, self.lin(h)) / deg)


class GATLayer(nn.Module):
    """Multi-head graph attention restricted to `adj` (dependency neighbours).
    If n_rel > 0 the dependency relation of edge (i,j) adds a learned bias to the
    attention logit and a relation embedding to the message."""

    def __init__(self, d_in, d_out, heads=4, n_rel=0, dropout=0.1, norm=True):
        super().__init__()
        self.use_norm = norm
        assert d_out % heads == 0
        self.h, self.dk = heads, d_out // heads
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dk))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dk))
        # zero init => uniform attention over dependency neighbours at the start,
        # i.e. the layer starts as GCN-style mean aggregation and learns to deviate
        nn.init.zeros_(self.a_src)
        nn.init.zeros_(self.a_dst)
        self.n_rel = n_rel
        if n_rel:
            self.rel_bias = nn.Embedding(n_rel, heads)
            self.rel_msg = nn.Embedding(n_rel, d_out)
            nn.init.zeros_(self.rel_bias.weight)
            nn.init.normal_(self.rel_msg.weight, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.res = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, h, adj, rel=None):
        B, N, _ = h.shape
        z = self.W(h).view(B, N, self.h, self.dk)                   # B N H dk
        s = (z * self.a_src).sum(-1)                                 # B N H
        d = (z * self.a_dst).sum(-1)
        logits = F.leaky_relu(s.unsqueeze(2) + d.unsqueeze(1), 0.2)  # B N(i) N(j) H
        if self.n_rel and rel is not None:
            logits = logits + self.rel_bias(rel)
        logits = logits.masked_fill(adj.unsqueeze(-1) == 0, -1e9)
        att = self.drop(torch.softmax(logits, dim=2))                # over neighbours j
        msg = torch.einsum("bijh,bjhd->bihd", att, z).reshape(B, N, -1)
        if self.n_rel and rel is not None:
            # relation messages: sum_j att_ij * r_ij  (averaged over heads)
            # computed as (attention mass per relation type) @ relation embeddings,
            # which avoids materialising a B x N x N x D tensor
            M = torch.zeros(B, N, self.n_rel, device=h.device, dtype=att.dtype)
            M.scatter_add_(2, rel, att.mean(-1))                      # B N R
            msg = msg + M @ self.rel_msg.weight
        if not self.use_norm:
            # same output form as GCNLayer (ReLU, no residual/LayerNorm) so the
            # ASGCN-style dot-product readout sees features on the same scale
            return F.relu(self.out(msg))
        return self.norm(F.elu(self.out(msg)) + self.res(h))


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


# ------------------------------------------------------------------ GloVe models
class GloveGraphModel(nn.Module):
    def __init__(self, emb_matrix, n_rel, kind="gcn", hidden=300, layers=2, heads=4,
                 dropout=0.5, n_class=3, n_pos=20, use_pos=True):
        super().__init__()
        self.kind = kind
        self.use_pos = use_pos
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float), freeze=True, padding_idx=0)
        self.pos_emb = nn.Embedding(n_pos, 30, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        d_emb = emb_matrix.shape[1] + 30
        self.lstm = nn.LSTM(d_emb, hidden, batch_first=True, bidirectional=True)
        d = 2 * hidden
        if kind == "gcn":
            self.gnn = nn.ModuleList([GCNLayer(d, d) for _ in range(layers)])
        elif kind in ("gat", "rgat"):
            self.gnn = nn.ModuleList([GATLayer(d, d, heads, n_rel if kind == "rgat" else 0, 0.1, norm=False)
                                      for _ in range(layers)])
        elif kind == "lstm":
            self.gnn = nn.ModuleList()
        else:
            raise ValueError(kind)
        self.gnn_drop = nn.Dropout(0.3)
        self.fc = nn.Linear(d, n_class)

    def forward(self, b):
        pe = self.pos_emb(b["postag"]) if self.use_pos else torch.zeros_like(self.pos_emb(b["postag"]))
        x = torch.cat([self.emb(b["tok"]), pe], -1)
        x = self.emb_drop(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, b["len"].cpu(), batch_first=True, enforce_sorted=False)
        H, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=x.size(1))
        g = H
        for layer in self.gnn:
            if self.kind == "gcn":
                g = layer(g * b["pw"].unsqueeze(-1), b["adj"])
            else:  # same ASGCN position weighting as the GCN for a fair comparison
                g = layer(g * b["pw"].unsqueeze(-1), b["adj"], b.get("rel"))
            g = self.gnn_drop(g)
        # aspect-specific readout (ASGCN): aspect vector from the graph output
        # attends over the BiLSTM states of the whole sentence
        asp = (g * b["amask"].unsqueeze(-1)).sum(1, keepdim=True)          # B 1 D
        scores = torch.bmm(asp, H.transpose(1, 2)).squeeze(1)               # B N
        scores = scores.masked_fill(b["mask"] == 0, -1e9)
        alpha = torch.softmax(scores, -1)
        r = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        self.last_attention = alpha.detach()
        return self.fc(r)


# ------------------------------------------------------------------ transformer models
class RobertaABSA(nn.Module):
    def __init__(self, plm_path, kind="roberta", n_rel=0, layers=2, heads=4, gnn_dim=384,
                 dropout=0.1, n_class=3):
        super().__init__()
        from transformers import AutoModel
        self.kind = kind
        self.enc = AutoModel.from_pretrained(plm_path, add_pooling_layer=False) \
            if "roberta" in plm_path or "spacy" in plm_path else AutoModel.from_pretrained(plm_path)
        d = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        if kind == "roberta":
            self.fc = nn.Linear(d, n_class)
        else:
            self.proj = nn.Linear(d, gnn_dim)
            if kind == "roberta-gcn":
                self.gnn = nn.ModuleList([GCNLayer(gnn_dim, gnn_dim) for _ in range(layers)])
            else:
                self.gnn = nn.ModuleList([GATLayer(gnn_dim, gnn_dim, heads, n_rel, 0.1) for _ in range(layers)])
            self.fc = nn.Linear(d + gnn_dim, n_class)

    def forward(self, b):
        out = self.enc(input_ids=b["ids"], attention_mask=b["att"]).last_hidden_state
        cls = out[:, 0]
        if self.kind == "roberta":
            return self.fc(self.drop(cls))
        # gather first sub-word of every word -> word-level states
        idx = b["first"].unsqueeze(-1).expand(-1, -1, out.size(-1))
        w = torch.gather(out, 1, idx)
        g = self.drop(F.relu(self.proj(w)))
        for layer in self.gnn:
            g = layer(g, b["adj"]) if self.kind == "roberta-gcn" else layer(g, b["adj"], b.get("rel"))
        asp = masked_mean(g, b["amask"])
        return self.fc(self.drop(torch.cat([cls, asp], -1)))


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
