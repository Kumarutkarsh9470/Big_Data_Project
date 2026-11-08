"""Export the RoBERTa-base encoder shipped inside spaCy's `en_core_web_trf`
pipeline to a HuggingFace-compatible folder (config + weights + tokenizer).

Why: our compute environment cannot reach the HuggingFace Hub, but GitHub
release assets (spaCy models) are reachable. `en_core_web_trf` contains a full
RoBERTa-base (125M params) whose weights were additionally fine-tuned by spaCy
on OntoNotes (tagging/parsing/NER). If you *do* have Hub access you can skip this
and pass `--plm roberta-base` to the training script instead.

Usage:  python scripts/export_roberta_from_spacy.py --out resources/roberta-base-spacy
"""
import argparse, json, os
import torch
import spacy
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resources/roberta-base-spacy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nlp = spacy.load("en_core_web_trf")
    tmodel = nlp.get_pipe("transformer").model
    proc, sd = None, None
    for node in tmodel.walk():
        if node.name == "byte_bpe_encoder":
            proc = node.attrs["byte_bpe_processor"]
        if node.name == "pytorch":
            sd = node.shims[0]._model.state_dict()
    assert proc is not None and sd is not None

    # ---- tokenizer: dump vocab + merges in GPT-2/RoBERTa format
    vocab = proc.vocab
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(args.out, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
        for a, b in proc.merges:
            f.write(f"{a} {b}\n")
    # transformers>=5 takes the vocab dict / merges list directly
    tok = RobertaTokenizer(vocab=dict(vocab), merges=[tuple(m) for m in proc.merges])
    tok.save_pretrained(args.out)

    # ---- weights: curated-transformers -> HF Roberta naming
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("curated_encoder.layers."))
    cfg = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=n_layers,
                        num_attention_heads=12, intermediate_size=3072,
                        max_position_embeddings=514, type_vocab_size=1,
                        layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    model = RobertaModel(cfg, add_pooling_layer=False)
    new = {}
    p = "curated_encoder."
    e = p + "embeddings.inner."
    new["embeddings.word_embeddings.weight"] = sd[e + "word_embeddings.weight"]
    new["embeddings.position_embeddings.weight"] = sd[e + "position_embeddings.weight"]
    new["embeddings.token_type_embeddings.weight"] = sd[e + "token_type_embeddings.weight"]
    new["embeddings.LayerNorm.weight"] = sd[e + "layer_norm.weight"]
    new["embeddings.LayerNorm.bias"] = sd[e + "layer_norm.bias"]
    for i in range(n_layers):
        L = f"{p}layers.{i}."
        H = f"encoder.layer.{i}."
        qkv_w, qkv_b = sd[L + "mha.input.weight"], sd[L + "mha.input.bias"]
        for j, name in enumerate(["query", "key", "value"]):
            new[H + f"attention.self.{name}.weight"] = qkv_w[j * 768:(j + 1) * 768]
            new[H + f"attention.self.{name}.bias"] = qkv_b[j * 768:(j + 1) * 768]
        new[H + "attention.output.dense.weight"] = sd[L + "mha.output.weight"]
        new[H + "attention.output.dense.bias"] = sd[L + "mha.output.bias"]
        new[H + "attention.output.LayerNorm.weight"] = sd[L + "attn_output_layernorm.weight"]
        new[H + "attention.output.LayerNorm.bias"] = sd[L + "attn_output_layernorm.bias"]
        new[H + "intermediate.dense.weight"] = sd[L + "ffn.intermediate.weight"]
        new[H + "intermediate.dense.bias"] = sd[L + "ffn.intermediate.bias"]
        new[H + "output.dense.weight"] = sd[L + "ffn.output.weight"]
        new[H + "output.dense.bias"] = sd[L + "ffn.output.bias"]
        new[H + "output.LayerNorm.weight"] = sd[L + "ffn_output_layernorm.weight"]
        new[H + "output.LayerNorm.bias"] = sd[L + "ffn_output_layernorm.bias"]
    missing, unexpected = model.load_state_dict(new, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    model.save_pretrained(args.out)
    print(f"saved RoBERTa ({n_layers} layers) + tokenizer to {args.out}")

    # sanity check: tokenizer ids identical to spaCy's processor
    s = "The battery life is great, but the screen is dim."
    assert tok(s, add_special_tokens=False)["input_ids"] == proc.encode(s)[0]
    print("tokenizer check ok")


if __name__ == "__main__":
    main()
