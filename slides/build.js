const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("./apply_theme");

const THEME = {
  name: "ABSA Syntax",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2933", lt1: "FFFFFF", dk2: "14213D", lt2: "F1F4F6",
    accent1: "2A9D8F", accent2: "E76F51", accent3: "F4A261", accent4: "264653",
    accent5: "8D99AE", accent6: "6C5B7B", hlink: "2A9D8F", folHlink: "6C5B7B",
  },
};
const H = THEME.colors; // hex, only for options that require hex
const OUT = "ABSA_GNN_vs_Transformer_slides.pptx";

async function icon(Comp, hex, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "Does Syntax Still Help? Dependency-tree GNNs vs Transformers for ABSA";
  const C = pres.SchemeColor;

  const ICON = {
    task: await icon(fa.FaCommentDots, H.lt1), problem: await icon(fa.FaRandom, H.lt1),
    idea: await icon(fa.FaProjectDiagram, H.lt1), q1: await icon(fa.FaBalanceScale, H.lt1),
    q2: await icon(fa.FaSitemap, H.lt1), q3: await icon(fa.FaTags, H.lt1), q4: await icon(fa.FaServer, H.lt1),
    check: await icon(fa.FaCheck, H.lt1), cross: await icon(fa.FaTimes, H.lt1),
    bolt: await icon(fa.FaBolt, H.lt1), warn: await icon(fa.FaExclamationTriangle, H.lt1),
    bulb: await icon(fa.FaLightbulb, H.lt1), chart: await icon(fa.FaChartBar, H.lt1),
    github: await icon(fa.FaCode, H.lt1),
  };

  // ---------- layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.55, w: 12.1, h: 1.1, fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "top", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.7, w: 11.5, h: 0.9, fontSize: 20, color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.5, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.75, fontSize: 30, bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "Dependency-tree GNNs vs Transformers for ABSA", options: { x: 0.6, y: 7.05, w: 7, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.75, fontSize: 30, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
  });

  // ---------- helpers
  const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });
  const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" }, objectName: name });
  const iconCircle = (s, img, x, y, d, fill, name = "icon") => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: name + " circle" });
    const p = d * 0.26;
    s.addImage({ data: img, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name });
  };
  const txt = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, color: C.text1, fontSize: 14, valign: "top", ...o });
  const arrow = (s, x, y, w, h, color = C.accent5) =>
    s.addShape(pres.shapes.LINE, { x, y, w, h, line: { color, width: 2, endArrowType: "triangle" }, objectName: "arrow" });
  const chartBase = (title) => ({
    showTitle: !!title, title, titleFontSize: 14, titleColor: H.dk1, titleFontFace: "+mj-lt",
    catAxisLabelColor: H.dk1, valAxisLabelColor: "6B7684", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E1E5EA", size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: true, valAxisLineShow: false,
    showValue: true, dataLabelFontSize: 10, dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd",
    showLegend: true, legendPos: "t", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: H.dk1,
  });

  // =====================================================================
  // 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Does Syntax Still Help?", { placeholder: "title" });
  s.addText("Dependency-tree GCN / GAT versus Transformers for aspect-based sentiment on Amazon Reviews 2023 and Yelp", { placeholder: "body" });
  s.addImage({ path: "assets/dep_dark.png", x: 0.6, y: 2.95, w: 12.1, h: 12.1 * 594 / 2210, objectName: "dependency tree example" });
  txt(s, [
    { text: "Author One  |  Author Two  |  Author Three", options: { bold: true, color: C.background1, breakLine: true } },
    { text: "Big Data course project", options: { color: C.accent5 } },
  ], { x: 0.6, y: 6.35, w: 8, h: 0.75, fontSize: 14 });
  txt(s, [
    { text: "positive aspect", options: { color: C.accent1, bold: true } },
    { text: "    negative aspect", options: { color: C.accent2, bold: true } },
  ], { x: 8.7, y: 6.55, w: 4, h: 0.35, fontSize: 14, align: "right" });
  s.addNotes("Our project asks whether dependency-tree graph neural networks, which were the standard way to add syntax to aspect-based sentiment classification, still help now that pre-trained transformers exist. The picture at the bottom is the running example: one sentence, two aspects, opposite opinions. The coloured arcs are the syntactic paths that connect each aspect to its opinion word.");

  // 2. Task + motivation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("One sentence, two aspects, opposite opinions", { placeholder: "title" });
  card(s, 0.6, 1.25, 12.1, 3.05, C.background2, "parse panel");
  s.addImage({ path: "assets/dep_light.png", x: 0.85, y: 1.4, w: 11.6, h: 11.6 * 594 / 2210, objectName: "dependency parse" });
  const cards2 = [
    [ICON.task, C.accent4, "The task", "Given a sentence and an aspect term, predict positive / negative / neutral for that aspect. Here: battery life is positive, screen is negative."],
    [ICON.problem, C.accent2, "The problem", "\"great\" and \"dim\" are both 3 words from \"screen\". Sequence models with attention often pick the strongest opinion word, not the right one."],
    [ICON.idea, C.accent1, "The idea", "In the parse, \"dim\" is 2 arcs from \"screen\" and \"great\" is 3. A GNN that passes messages along arcs should attach the right opinion."],
  ];
  cards2.forEach(([ic, col, head, body], i) => {
    const x = 0.6 + i * 4.1;
    card(s, x, 4.55, 3.9, 2.3, C.background2, head + " card");
    iconCircle(s, ic, x + 0.25, 4.75, 0.6, col, head + " icon");
    txt(s, head, { x: x + 1.0, y: 4.83, w: 2.7, h: 0.45, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
    txt(s, body, { x: x + 0.25, y: 5.5, w: 3.45, h: 1.25, fontSize: 14 });
  });
  s.addNotes("Aspect-based sentiment classification gives a polarity per aspect, not per sentence. In this sentence the two aspects disagree. Measured in words, both opinion words are three tokens away from screen, so a sequence model has no positional cue. In the dependency tree, dim hangs off the clause whose subject is screen, two arcs away, while great is three arcs away through the conjunction. That is the motivation for GCN and GAT models over the parse.");

  // 3. Research questions and setup
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Four questions, one controlled setup", { placeholder: "title" });
  const qs = [
    [ICON.q1, C.accent4, "Graph vs transformer", "Do GCN / GAT over the parse beat a BiLSTM, and how far are they from fine-tuned RoBERTa?"],
    [ICON.q2, C.accent1, "Is it really syntax?", "Replace the parse with a word chain, a random tree, a complete graph. Does accuracy drop?"],
    [ICON.q3, C.accent2, "Labels from star ratings?", "Amazon has millions of reviews but no aspect labels. Can ratings stand in for them?"],
    [ICON.q4, C.accent6, "Cost at scale", "What does the accuracy of a transformer cost when tagging 100k reviews on a CPU?"],
  ];
  qs.forEach(([ic, col, head, body], i) => {
    const y = 1.3 + i * 1.38;
    iconCircle(s, ic, 0.6, y + 0.05, 0.75, col, head);
    txt(s, `RQ${i + 1}  ${head}`, { x: 1.6, y, w: 6.4, h: 0.45, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
    txt(s, body, { x: 1.6, y: y + 0.48, w: 6.4, h: 0.75, fontSize: 14 });
  });
  card(s, 8.5, 1.3, 4.2, 5.4, C.text2, "setup panel");
  const stats = [["7", "models (4 GloVe, 3 RoBERTa)"], ["6", "test sets, 2 manually labelled for Amazon"], ["137", "training runs, 3 seeds for GloVe"], ["0", "test-set peeking: selection on dev only"]];
  stats.forEach(([n, l], i) => {
    const y = 1.55 + i * 1.27;
    txt(s, n, { x: 8.75, y, w: 1.25, h: 0.8, fontSize: 40, align: "right", bold: true, color: C.accent3, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, l, { x: 10.25, y: y + 0.08, w: 2.25, h: 0.7, fontSize: 14, color: C.background1, valign: "middle" });
  });
  s.addNotes("We organised the work around four questions. First, the head-to-head comparison. Second, an ablation that checks whether any gain really comes from syntax or just from having a sparse graph. Third, whether star ratings can replace aspect labels on Amazon. Fourth, the throughput cost of transformers at corpus scale. Everything shares one training protocol: same parser, same splits, model selection only on a development split, and several seeds.");

  // 4. Data
  pres.addSection({ title: "Data" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
  s.addText("Six test sets, from benchmarks to Amazon and Yelp", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 13 } });
  const rows = [
    [hdr("Dataset"), hdr("Domain"), hdr("Labels"), hdr("Train"), hdr("Test"), hdr("Mixed %")],
    ["Rest14", "restaurants", "SemEval-14", "3,246", "1,120", "19.5"],
    ["Lap14", "laptops", "SemEval-14", "2,095", "638", "15.2"],
    ["MAMS", "restaurants", "human", "11,186", "1,336", "99.9"],
    ["Yelp", "Yelp reviews", "human", "727", "245", "9.0"],
    ["Amz-Dist", "Amazon 2023 Electronics", "star rating + VADER", "6,805", "858", "0.0"],
    ["Amz-Gold", "Amazon 2023 Electronics", "manual", "n/a", "225", "41.3"],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.3, w: 7.3, colW: [1.15, 2.05, 1.6, 0.85, 0.75, 0.9], fontSize: 13, color: C.text1,
    border: { type: "solid", pt: 0.5, color: "D5DAE0" }, rowH: 0.48, valign: "middle", objectName: "dataset table",
  });
  txt(s, "Mixed % = share of test aspects whose sentence also contains an aspect with a different polarity. Long Yelp reviews are cropped to the aspect's sentence; every example is parsed with spaCy en_core_web_lg.", { x: 0.6, y: 4.85, w: 7.3, h: 0.9, fontSize: 12, color: C.accent5 });
  card(s, 0.6, 5.85, 7.3, 0.95, C.background2, "yelp note");
  txt(s, [{ text: "Yelp is small and skewed: ", options: { bold: true } }, { text: "727 training aspects, 64% positive, only 49 neutral. This matters for RoBERTa later." }], { x: 0.85, y: 6.0, w: 6.9, h: 0.7, fontSize: 14, valign: "middle" });
  s.addChart(pres.charts.BAR, [
    { name: "negative", labels: ["Rest14", "Lap14", "MAMS", "Yelp", "Amz-Dist", "Amz-Gold"], values: [17.5, 20.1, 24.6, 20.4, 37.3, 16.0] },
    { name: "neutral", labels: ["Rest14", "Lap14", "MAMS", "Yelp", "Amz-Dist", "Amz-Gold"], values: [17.5, 26.5, 45.4, 6.5, 21.8, 55.6] },
    { name: "positive", labels: ["Rest14", "Lap14", "MAMS", "Yelp", "Amz-Dist", "Amz-Gold"], values: [65.0, 53.4, 29.9, 73.1, 40.9, 28.4] },
  ], {
    ...chartBase("Test label distribution (%)"), x: 8.2, y: 1.2, w: 4.6, h: 5.6, barDir: "bar", barGrouping: "percentStacked",
    chartColors: [H.accent2, "7B8794", H.accent1], dataLabelPosition: "ctr", dataLabelColor: "FFFFFF", dataLabelFormatCode: "0",
    valAxisHidden: true, catAxisOrientation: "maxMin", barGapWidthPct: 45, objectName: "label distribution chart",
  });
  s.addNotes("Rest14 and Lap14 are the standard SemEval benchmarks; restaurants are the closest domain to Yelp and laptops to Amazon Electronics. MAMS is a stress test where nearly every sentence has aspects with different polarities. The Yelp set is a human-labelled Yelp ABSA set, but it is small and mostly positive. For Amazon Reviews 2023 we built two sets ourselves, described on the next slide. Note how different the Amazon gold set is: more than half neutral and 41 percent of aspects in mixed sentences.");

  // 5. Amazon pipeline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data" });
  s.addText("Building aspect-level data from Amazon Reviews 2023", { placeholder: "title" });
  const steps = [
    ["100k reviews", "Electronics, star ratings only"],
    ["161k sentences", "spaCy split + dependency parse"],
    ["Aspect lexicon", "400 nouns + 200 compounds (battery life, sound quality)"],
    ["Distant labels", "rating >= 4 and VADER >= 0.5: pos\nrating <= 2 and VADER <= -0.3: neg\nno sentiment words: neutral"],
  ];
  steps.forEach(([h1, b], i) => {
    const x = 0.6 + i * 2.55;
    card(s, x, 1.3, 2.25, 2.35, i === 3 ? C.accent4 : C.background2, h1);
    txt(s, h1, { x: x + 0.15, y: 1.42, w: 1.95, h: 0.45, fontSize: 16, bold: true, color: i === 3 ? C.background1 : C.text2, fontFace: THEME.headFontFace });
    txt(s, b, { x: x + 0.15, y: 1.92, w: 1.98, h: 1.65, fontSize: i === 3 ? 12 : 14, color: i === 3 ? C.background1 : C.text1 });
    if (i < 3) arrow(s, x + 2.27, 2.47, 0.26, 0);
  });
  arrow(s, 9.0, 3.68, 0, 0.55);
  arrow(s, 1.7, 3.68, 0, 0.55, C.accent2);
  card(s, 7.7, 4.3, 2.6, 1.95, C.background2, "amz-dist card");
  txt(s, [{ text: "Amz-Dist", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } }, { text: "8,500 aspects, class-capped, split by review. Contrastive sentences removed.", options: { fontSize: 14 } }], { x: 7.9, y: 4.45, w: 2.25, h: 2.2 });
  card(s, 0.6, 4.3, 6.8, 1.95, C.background2, "amz-gold card");
  txt(s, [
    { text: "Amz-Gold: 3k held-out reviews", options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
    { text: "260 sampled candidates, over-sampling sentences with \"but / however\" and 2+ aspects. Labelled per aspect with SemEval-2014 guidelines.", options: { fontSize: 14, breakLine: true } },
    { text: "225 valid aspects: 64 pos / 36 neg / 125 neu, two thirds from contrastive sentences.", options: { fontSize: 14, bold: true, color: C.accent2 } },
  ], { x: 0.85, y: 4.45, w: 6.35, h: 2.2 });
  s.addChart(pres.charts.DOUGHNUT, [{ name: "Audit", labels: ["correct", "wrong label", "not an aspect"], values: [62, 27, 11] }], {
    x: 10.55, y: 1.15, w: 2.3, h: 2.6, holeSize: 58, chartColors: [H.accent1, H.accent2, "C3CAD3"],
    showLegend: true, legendPos: "b", legendFontSize: 10, legendFontFace: "+mn-lt", legendColor: H.dk1,
    showValue: false, showPercent: false, showLabel: false, showTitle: false, dataLabelColor: "FFFFFF", objectName: "audit doughnut",
  });
  txt(s, [{ text: "70%", options: { fontSize: 36, bold: true, color: C.accent2, fontFace: THEME.headFontFace, breakLine: true } },
          { text: "of distant labels correct on a 100-item manual audit (valid aspects)", options: { fontSize: 13 } }], { x: 10.55, y: 4.3, w: 2.2, h: 2.4 });
  s.addNotes("Amazon Reviews 2023 has ratings but no aspect labels. We parse 40 thousand reviews, mine an aspect lexicon of frequent nouns and compound nouns, and label sentences only when the star rating and the VADER sentence score agree. Contrastive sentences are dropped because their aspects can disagree. To know how good these labels are, we hand-checked 100 of them: 11 candidates are not aspects at all, and only 70 percent of the rest are correct, almost always because an incidental noun inherits the sentence's sentiment. For evaluation we built a separate manually labelled set from held-out reviews that focuses on exactly the hard contrastive sentences.");

  // 6. Models
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Seven models: only the encoder and the graph layer change", { placeholder: "title" });
  // input
  card(s, 0.6, 2.6, 1.9, 1.9, C.background2, "input");
  txt(s, [{ text: "Input", options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } }, { text: "sentence + aspect span + dependency parse", options: { fontSize: 13 } }], { x: 0.75, y: 2.75, w: 1.65, h: 1.65 });
  arrow(s, 2.55, 2.85, 0.6, -0.95);
  arrow(s, 2.55, 4.25, 0.6, 0.95);
  // encoders
  card(s, 3.2, 1.3, 3.0, 1.55, C.accent5, "glove encoder");
  txt(s, [{ text: "GloVe 300d + POS", options: { bold: true, fontSize: 16, color: C.background1, breakLine: true } }, { text: "BiLSTM, 2 x 300 units, embeddings frozen", options: { fontSize: 13, color: C.background1 } }], { x: 3.4, y: 1.45, w: 2.7, h: 1.3 });
  card(s, 3.2, 4.25, 3.0, 1.55, C.accent3, "roberta encoder");
  txt(s, [{ text: "RoBERTa-base", options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } }, { text: "fine-tuned, first sub-word per word, 125M params", options: { fontSize: 13, color: C.text2 } }], { x: 3.4, y: 4.4, w: 2.7, h: 1.3 });
  arrow(s, 6.25, 2.1, 0.6, 0.9);
  arrow(s, 6.25, 5.0, 0.6, -0.9);
  // graph layer
  card(s, 6.9, 1.85, 3.0, 3.45, C.text2, "graph layer");
  txt(s, "Graph layer (x2)", { x: 7.1, y: 2.0, w: 2.6, h: 0.4, fontSize: 16, bold: true, color: C.background1 });
  const chips = [["none", "BiLSTM+Att, RoBERTa"], ["GCN", "mean over neighbours"], ["GAT", "4-head attention"], ["R-GAT", "+ relation labels"]];
  chips.forEach(([a, b], i) => {
    const y = 2.5 + i * 0.68;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.1, y, w: 2.6, h: 0.56, rectRadius: 0.08, fill: { color: i === 0 ? C.accent5 : C.accent1 }, line: { type: "none" }, objectName: "chip " + a });
    txt(s, [{ text: a + "  ", options: { bold: true } }, { text: b }], { x: 7.22, y, w: 2.4, h: 0.56, fontSize: 13, color: C.background1, valign: "middle" });
  });
  arrow(s, 9.95, 3.55, 0.45, 0);
  // readout
  card(s, 10.45, 2.6, 2.25, 1.9, C.background2, "readout");
  txt(s, [{ text: "Aspect readout", options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } }, { text: "pool aspect nodes, attend / concat with <s>, softmax over 3 classes", options: { fontSize: 13 } }], { x: 10.6, y: 2.75, w: 2.0, h: 1.65 });
  // protocol strip
  const prot = [["Same input", "one parser, one set of splits"], ["Dev-only selection", "best dev epoch, test used once"], ["Seeds", "3 for GloVe, 1-2 for RoBERTa"], ["Significance", "paired bootstrap, 5k resamples"]];
  prot.forEach(([a, b], i) => {
    const x = 0.6 + i * 3.075;
    card(s, x, 6.05, 2.875, 0.85, C.background2, "protocol " + a);
    txt(s, [{ text: a, options: { bold: true, color: C.text2, breakLine: true } }, { text: b, options: { fontSize: 12 } }], { x: x + 0.15, y: 6.1, w: 2.6, h: 0.75, fontSize: 14, valign: "middle" });
  });
  s.addNotes("All seven models share one pipeline. The encoder is either GloVe with a BiLSTM or a fine-tuned RoBERTa-base. The graph layer is either absent, a GCN in the ASGCN style, a four-head graph attention network, or a relation-aware GAT that also uses the dependency labels. The readout pools the aspect positions. One note on the GAT: our first version had an extra projection and layer norm and trained badly; we fixed it so it has the same form as the GCN, and all numbers use the fixed version. RoBERTa weights were taken from spaCy's transformer pipeline because the Hugging Face hub was not reachable from our machine.");

  // 7. Main results
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("The pre-trained encoder matters far more than the graph", { placeholder: "title" });
  const cats = ["Rest14", "Lap14", "MAMS", "Yelp", "Lap14 to Amz-Gold"];
  s.addChart(pres.charts.BAR, [
    { name: "BiLSTM (no graph)", labels: cats, values: [78.96, 70.59, 77.97, 84.35, 51.26] },
    { name: "GCN", labels: cats, values: [79.08, 69.28, 78.44, 84.76, 48.0] },
    { name: "R-GAT", labels: cats, values: [79.14, 70.48, 78.47, 84.49, 46.07] },
    { name: "RoBERTa", labels: cats, values: [84.02, 77.66, 83.46, 81.02, 59.11] },
    { name: "RoBERTa+R-GAT", labels: cats, values: [84.11, 78.13, 83.23, 83.67, 63.11] },
  ], {
    ...chartBase("Test accuracy (%)"), x: 0.6, y: 1.2, w: 8.6, h: 5.65, barDir: "col", barGrouping: "clustered",
    chartColors: ["C3CAD3", H.accent5, H.accent4, H.accent3, H.accent1], valAxisMinVal: 40, valAxisMaxVal: 90,
    valAxisMajorUnit: 10, barGapWidthPct: 55, dataLabelFontSize: 9, dataLabelFormatCode: "0", objectName: "main accuracy chart",
  });
  const callouts = [
    ["+5 to 8", "points: RoBERTa over the best GloVe model on Rest14, Lap14, MAMS", C.accent3],
    ["1.8", "points: largest gap between the four GloVe models on any benchmark (seed noise level)", C.accent4],
    ["+4 / +5", "points on Amz-Gold when a dependency GNN is added to RoBERTa (p = 0.07 / 0.02)", C.accent1],
  ];
  callouts.forEach(([n, l, col], i) => {
    const y = 1.3 + i * 1.85;
    card(s, 9.5, y, 3.2, 1.65, C.background2, "callout " + i);
    txt(s, n, { x: 9.7, y: y + 0.1, w: 2.9, h: 0.65, fontSize: 32, bold: true, color: col, fontFace: THEME.headFontFace });
    txt(s, l, { x: 9.7, y: y + 0.78, w: 2.85, h: 0.8, fontSize: 13 });
  });
  s.addNotes("Main results. RoBERTa beats every GloVe model by five to eight points on the benchmarks. The four GloVe models are within 1.8 points of each other, which is the size of the seed variation, and none of the graph-versus-BiLSTM differences is significant. On top of RoBERTa the graph adds little in-domain, but on the Amazon gold set the hybrids gain four to five points. Yelp is the odd one out: plain RoBERTa scores below the GloVe models because with only 49 neutral training examples it never predicts neutral; the hybrids recover.");

  // 8. Ablation
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Is it the syntax, or just having a graph?", { placeholder: "title" });
  const gcats = ["Dependency tree", "Word chain", "Random tree", "Complete graph"];
  s.addChart(pres.charts.BAR, [
    { name: "GCN", labels: gcats, values: [79.08, 80.33, 76.55, 77.92] },
    { name: "GAT", labels: gcats, values: [78.51, 79.88, 76.61, 78.54] },
  ], {
    ...chartBase("Rest14 test accuracy (%), GloVe encoder, 3 seeds"), x: 0.6, y: 1.2, w: 7.4, h: 5.0, barDir: "col",
    chartColors: [H.accent4, H.accent1], valAxisMinVal: 74, valAxisMaxVal: 82, valAxisMajorUnit: 2, barGapWidthPct: 60,
    objectName: "graph ablation chart",
  });
  txt(s, "BiLSTM without any graph: 79.0. Same ranking on Lap14, but there all graphs are within seed noise (68.0 to 70.5).", { x: 0.6, y: 6.3, w: 7.4, h: 0.6, fontSize: 12, color: C.accent5 });
  const abl = [
    [ICON.check, C.accent1, "The parse is not arbitrary", "A random tree with the same number of edges loses 2.5 points of accuracy and 5.2 of macro-F1."],
    [ICON.cross, C.accent2, "But a word chain is as good", "Linking only adjacent words scores 80.3, higher than the parse. Depth (1 to 4 layers) changes less than 1.3."],
    [ICON.bulb, C.accent3, "Reading", "With a BiLSTM encoder the graph mainly adds local smoothing. Most dependency arcs are short anyway."],
  ];
  abl.forEach(([ic, col, h1, b], i) => {
    const y = 1.3 + i * 1.85;
    iconCircle(s, ic, 8.4, y + 0.05, 0.6, col, h1);
    txt(s, h1, { x: 9.2, y, w: 3.5, h: 0.45, fontSize: 17, bold: true, color: C.text2, fontFace: THEME.headFontFace });
    txt(s, b, { x: 9.2, y: y + 0.5, w: 3.5, h: 1.2, fontSize: 14 });
  });
  s.addNotes("To check whether gains come from syntax, we kept everything fixed and swapped the graph. A random tree of the same size is clearly worse than the parse, so the parse carries information. But a plain chain that links neighbouring words does as well or better. Since the BiLSTM already encodes word order, what the graph contributes here is mostly locality. Varying the number of GCN layers from one to four made no real difference.");

  // 9. Where syntax helps
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Syntax helps a transformer where aspects disagree", { placeholder: "title" });
  const mcats = ["Lap14 (n=97)", "Amz-Gold (n=93)", "Rest14 (n=218)"];
  s.addChart(pres.charts.BAR, [
    { name: "RoBERTa", labels: mcats, values: [44.3, 46.8, 67.2] },
    { name: "RoBERTa+GCN", labels: mcats, values: [59.8, 59.1, 68.8] },
    { name: "RoBERTa+R-GAT", labels: mcats, values: [61.9, 62.4, 69.7] },
  ], {
    ...chartBase("Accuracy on mixed-polarity sentences (%), mean over seeds"), x: 0.6, y: 1.2, w: 7.6, h: 5.65, barDir: "col",
    chartColors: [H.accent3, H.accent1, H.accent4], valAxisMinVal: 30, valAxisMaxVal: 80, valAxisMajorUnit: 10, barGapWidthPct: 50,
    objectName: "mixed polarity chart",
  });
  card(s, 8.5, 1.3, 4.2, 1.9, C.text2, "headline");
  txt(s, [{ text: "+15 to +18", options: { fontSize: 40, bold: true, color: C.accent3, fontFace: THEME.headFontFace, breakLine: true } },
          { text: "points on Lap14 mixed sentences when a dependency GNN is added", options: { fontSize: 14, color: C.background1 } }], { x: 8.75, y: 1.45, w: 3.8, h: 1.65 });
  const pts = [
    [ICON.check, C.accent1, "Holds per seed: RoBERTa 42 / 46 vs hybrids 57 to 65 (Lap14); 44 / 50 vs 59 to 65 (Amz-Gold)"],
    [ICON.warn, C.accent3, "Rest14: small gain that depends on the seed (one RoBERTa seed scores 70.2)"],
    [ICON.cross, C.accent2, "MAMS: no gain (83.4 vs 83.1 / 83.2). Its 11k mixed training sentences teach RoBERTa the links itself"],
  ];
  pts.forEach(([ic, col, t], i) => {
    const y = 3.45 + i * 1.15;
    iconCircle(s, ic, 8.5, y + 0.05, 0.5, col, "point " + i);
    txt(s, t, { x: 9.15, y, w: 3.55, h: 1.05, fontSize: 14 });
  });
  s.addNotes("The averages hide the real effect. If we look only at aspects in sentences where another aspect has a different polarity, adding a dependency GNN to RoBERTa raises accuracy by 15 to 18 points on Lap14 and 12 to 16 on the Amazon gold set. This is not a lucky seed: both RoBERTa seeds are in the forties and all hybrid runs are between 57 and 65. On Rest14 the gain is small and seed-dependent, and on MAMS there is none. Rest14 has twice as many mixed training sentences as Lap14 and MAMS has over eleven thousand, so our reading is that the parse helps most when such examples are rare in training.");

  // 10. Distance + qualitative
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Distance to the opinion word, and what the errors look like", { placeholder: "title" });
  s.addImage({ path: "assets/distance_lap14.png", x: 0.6, y: 1.15, w: 7.6, h: 7.6 * 720 / 2000, objectName: "distance plot" });
  txt(s, "Lap14. GloVe models lose about 20 points between 1 and 3 arcs, transformers about 5. At 3 arcs the hybrids lead RoBERTa by 2 to 3 points.", { x: 0.6, y: 4.0, w: 7.6, h: 0.65, fontSize: 13, color: C.accent5 });
  const ex = [
    ["The battery life is great but the screen is too dim.", [["battery life", "pos", "pos", "pos"], ["screen", "neg", "pos", "neg"]]],
    ["The sound quality is amazing, although the ear cushions get uncomfortable.", [["sound quality", "pos", "pos", "pos"], ["ear cushions", "neg", "pos", "neg"]]],
    ["Customer service never answered, but the keyboard works fine.", [["customer service", "neg", "neu", "pos"], ["keyboard", "pos", "pos", "pos"]]],
  ];
  txt(s, [{ text: "aspect", options: { bold: true } }], { x: 8.55, y: 1.2, w: 1.7, h: 0.3, fontSize: 12, color: C.accent5 });
  txt(s, "gold", { x: 10.3, y: 1.2, w: 0.7, h: 0.3, fontSize: 12, bold: true, color: C.accent5, align: "center" });
  txt(s, "GloVe R-GAT", { x: 11.0, y: 1.2, w: 0.85, h: 0.3, fontSize: 10, bold: true, color: C.accent5, align: "center" });
  txt(s, "RoBERTa +GCN", { x: 11.85, y: 1.2, w: 0.85, h: 0.3, fontSize: 10, bold: true, color: C.accent5, align: "center" });
  const pc = { pos: C.accent1, neg: C.accent2, neu: C.accent5 };
  ex.forEach(([sent, asps], i) => {
    const y = 1.6 + i * 1.75;
    card(s, 8.45, y, 4.25, 1.6, C.background2, "example " + i);
    txt(s, sent, { x: 8.6, y: y + 0.08, w: 4.0, h: 0.62, fontSize: 12, italic: true });
    asps.forEach(([a, g, p1, p2], j) => {
      const yy = y + 0.75 + j * 0.4;
      txt(s, a, { x: 8.6, y: yy, w: 1.7, h: 0.34, fontSize: 13, bold: true, valign: "middle" });
      [[g, 10.35], [p1, 11.1], [p2, 11.95]].forEach(([lab, x], k) => {
        const wrong = k > 0 && lab !== g;
        s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy + 0.02, w: 0.65, h: 0.3, rectRadius: 0.06, fill: { color: pc[lab] }, line: wrong ? { color: C.text1, width: 2, dashType: "dash" } : { type: "none" }, objectName: "chip" });
        txt(s, lab, { x, y: yy + 0.02, w: 0.65, h: 0.3, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle" });
      });
    });
  });
  txt(s, "Dashed outline = wrong prediction. In the 2nd example the GloVe R-GAT puts 99% of its attention for \"ear cushions\" on \"amazing\".", { x: 0.6, y: 4.85, w: 7.6, h: 0.65, fontSize: 13 });
  card(s, 0.6, 5.65, 7.6, 1.2, C.background2, "takeaway");
  iconCircle(s, ICON.bulb, 0.8, 5.95, 0.6, C.accent3, "takeaway icon");
  txt(s, "GloVe graph models are pulled to the strongest opinion word of the sentence. On Lap14 mixed sentences they are 6 to 12 points below the plain BiLSTM.", { x: 1.6, y: 5.75, w: 6.45, h: 1.0, fontSize: 14, valign: "middle" });
  s.addNotes("Accuracy drops for all models as the opinion word moves further from the aspect in the tree, but the GloVe models drop much faster. On the right are real predictions from our saved models. The GloVe R-GAT gives both aspects the polarity of the strongest opinion word; in the ear-cushions example it puts 99 percent of its attention on amazing. RoBERTa with a GCN gets the first two sentences right but still fails on 'customer service never answered', which needs world knowledge rather than syntax.");

  // 11. Distant supervision
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Star-rating labels look great on paper and fail on real aspects", { placeholder: "title" });
  const dcats = ["Amz-Dist to Amz-Dist", "Amz-Dist to Amz-Gold", "Lap14 to Amz-Gold"];
  s.addChart(pres.charts.BAR, [
    { name: "BiLSTM", labels: dcats, values: [93.36, 49.48, 51.26] },
    { name: "RoBERTa", labels: dcats, values: [95.69, 51.11, 59.11] },
    { name: "RoBERTa+GCN", labels: dcats, values: [96.27, 54.22, 64.22] },
  ], {
    ...chartBase("Accuracy (%), train set to test set"), x: 0.6, y: 1.2, w: 7.6, h: 5.65, barDir: "col",
    chartColors: ["C3CAD3", H.accent3, H.accent1], valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, barGapWidthPct: 50,
    objectName: "distant supervision chart",
  });
  const big = [["96%", "on a test set labelled the same way as training", C.accent4], ["54%", "on the manually labelled Amazon set. Always predicting neutral gives 55.6%", C.accent2], ["64%", "from 2k human-labelled laptop sentences, a different domain", C.accent1]];
  big.forEach(([n, l, col], i) => {
    const y = 1.3 + i * 1.55;
    txt(s, n, { x: 8.55, y, w: 1.75, h: 1.0, fontSize: 40, bold: true, color: col, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, l, { x: 10.35, y: y + 0.05, w: 2.35, h: 1.0, fontSize: 14, valign: "middle" });
  });
  card(s, 8.5, 6.0, 4.2, 0.85, C.background2, "yelp transfer");
  txt(s, "Same on Yelp: RoBERTa trained on Rest14 scores 88.0 on Yelp, trained on Yelp 81.0.", { x: 8.65, y: 6.05, w: 3.95, h: 0.75, fontSize: 13, valign: "middle" });
  s.addNotes("Models trained on the distant Amazon labels reach 93 to 96 percent on a test set labelled the same way, far above the 70 percent label accuracy we measured. They have learnt the labelling rule. On the manually labelled set they fall to around 50 percent, below a constant neutral prediction in accuracy. A model trained on two thousand human-labelled laptop sentences does much better. The same pattern shows up for Yelp: training on the larger restaurant benchmark beats training on the small Yelp set itself. Correct aspect-level labels matter more than being in-domain.");

  // 12. Scale
  pres.addSection({ title: "Scale" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Scale" });
  s.addText("Tagging 97k Amazon reviews on a 4-core CPU", { placeholder: "title" });
  const sc = [["97,487", "reviews"], ["499,928", "sentences parsed"], ["588,417", "aspect mentions"], ["7.3 min", "end to end on CPU"]];
  sc.forEach(([n, l], i) => {
    const x = 0.6 + i * 1.9;
    txt(s, n, { x, y: 1.25, w: 1.85, h: 0.6, fontSize: 26, bold: true, color: C.accent4, fontFace: THEME.headFontFace });
    txt(s, l, { x, y: 1.85, w: 1.85, h: 0.4, fontSize: 13, color: C.accent5 });
  });
  s.addChart(pres.charts.BAR, [{ name: "mentions per second", labels: ["GloVe R-GAT", "RoBERTa+GCN"], values: [2207, 77] }], {
    ...chartBase("Inference throughput (mentions / s)"), x: 0.6, y: 2.45, w: 3.9, h: 3.0, barDir: "bar", showLegend: false,
    chartColors: [H.accent4, H.accent1], dataLabelFormatCode: "#,##0", valAxisHidden: true, barGapWidthPct: 40,
    catAxisOrientation: "maxMin", objectName: "throughput chart",
  });
  const cmp = [["Accuracy on Amz-Gold", "48.9%", "64.2%"], ["Spearman with stars (same 10k)", "0.39", "0.48"], ["Saw star ratings in training", "yes", "no"]];
  txt(s, "GloVe R-GAT", { x: 6.1, y: 2.55, w: 0.95, h: 0.35, fontSize: 12, bold: true, color: C.accent4, align: "center" });
  txt(s, "RoBERTa +GCN", { x: 7.05, y: 2.55, w: 0.95, h: 0.35, fontSize: 12, bold: true, color: C.accent1, align: "center" });
  cmp.forEach(([a, b, c], i) => {
    const y = 3.0 + i * 0.8;
    card(s, 4.7, y, 3.35, 0.68, C.background2, "compare " + i);
    txt(s, a, { x: 4.82, y, w: 1.3, h: 0.68, fontSize: 12, valign: "middle" });
    txt(s, b, { x: 6.1, y, w: 0.95, h: 0.68, fontSize: 15, bold: true, align: "center", valign: "middle", color: C.accent4 });
    txt(s, c, { x: 7.05, y, w: 0.95, h: 0.68, fontSize: 15, bold: true, align: "center", valign: "middle", color: C.accent1 });
  });
  card(s, 0.6, 5.7, 7.45, 1.15, C.text2, "tradeoff");
  iconCircle(s, ICON.bolt, 0.8, 5.97, 0.6, C.accent3, "bolt");
  txt(s, "28x slower, much more accurate, and the parse is almost free (spaCy parses 630 reviews/s). The two models agree on 62% of mentions.", { x: 1.6, y: 5.8, w: 6.3, h: 0.95, fontSize: 14, color: C.background1, valign: "middle" });
  s.addImage({ path: "assets/top_aspects.png", x: 8.35, y: 1.35, w: 4.35, h: 4.35 * 1600 / 1400, objectName: "top aspects chart" });
  s.addNotes("To look at cost we ran the whole pipeline over the 97 thousand Amazon reviews: about half a million sentences and 588 thousand aspect mentions in seven minutes with the GloVe R-GAT. RoBERTa with a GCN processes 77 mentions per second, 28 times slower, but it is far more accurate on the gold set, and its review-level aspect sentiment correlates better with star ratings even though it never saw a rating. The right panel shows the aggregate result: value, price and quality are the most positive aspects, battery and software the most negative.");

  // 13. Conclusions
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("Takeaways", { placeholder: "title" });
  const tk = [
    [ICON.chart, C.accent3, "Encoder first", "RoBERTa beats every GloVe graph model by 5 to 8 points. With GloVe, the parse adds no reliable gain over a BiLSTM and a word chain does as well."],
    [ICON.idea, C.accent1, "Syntax as a prior", "On top of RoBERTa, a dependency GNN fixes mixed-polarity sentences (+15 to 18 pts on Lap14, +12 to 16 on Amazon) when such training data is scarce."],
    [ICON.q3, C.accent2, "Check your labels", "Rating-based aspect labels are 70% correct; models trained on them score 96% on their own test and 54% on real aspects."],
    [ICON.bolt, C.accent6, "Pick by budget", "GloVe graph models tag 2,200 mentions/s on CPU; RoBERTa+GCN 77/s but 15 points more accurate on Amz-Gold."],
  ];
  tk.forEach(([ic, col, h1, b], i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.4 + Math.floor(i / 2) * 2.2;
    card(s, x, y, 5.9, 1.95, C.accent4, h1);
    iconCircle(s, ic, x + 0.25, y + 0.25, 0.65, col, h1);
    txt(s, h1, { x: x + 1.1, y: y + 0.3, w: 4.6, h: 0.5, fontSize: 20, bold: true, color: C.background1, fontFace: THEME.headFontFace });
    txt(s, b, { x: x + 1.1, y: y + 0.85, w: 4.6, h: 1.15, fontSize: 14, color: C.background2 });
  });
  txt(s, [
    { text: "Limitations: ", options: { bold: true, color: C.accent3 } },
    { text: "CPU budget (1 to 2 seeds for RoBERTa), RoBERTa weights from spaCy's en_core_web_trf, Amazon gold labelled in one LLM-assisted pass without agreement study, 100k-review sample of one category.", options: { color: C.background2 } },
  ], { x: 0.6, y: 6.15, w: 9.4, h: 0.8, fontSize: 13 });
  iconCircle(s, ICON.github, 10.2, 6.2, 0.5, C.accent1, "code icon");
  txt(s, [{ text: "Code, data, notebook", options: { bold: true, color: C.background1, breakLine: true } }, { text: "github.com/kumarutkarsh9470/big_data_course_project", options: { color: C.accent5, fontSize: 11 } }], { x: 10.8, y: 6.15, w: 2.4, h: 0.8, fontSize: 13 });
  s.addNotes("To sum up. The pre-trained encoder matters most, and with static embeddings the dependency graph is not worth the parsing. Syntax still has value as a prior on top of a transformer, specifically for sentences whose aspects disagree, when the training data has few of them. Star-rating labels are a trap unless you check them against manual labels. And at scale there is a clear trade-off between the fast GloVe graph models and the accurate transformer hybrids. The main limitations are the CPU budget, which limited transformer seeds, and the single-pass annotation of the Amazon gold set. Thank you.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
