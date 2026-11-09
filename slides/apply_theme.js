// Write the deck's color palette into ppt/theme/theme1.xml (pptxgenjs always writes the Office palette).
const fs = require("fs");
const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));

const SLOTS = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];

async function applyTheme(deckPath, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(deckPath));
  const part = "ppt/theme/theme1.xml";
  const xml = await zip.file(part).async("string");
  const name = theme.name.replace(/[&<>"]/g, "");
  const scheme = `<a:clrScheme name="${name}">` +
    SLOTS.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k].toUpperCase()}"/></a:${k}>`).join("") +
    "</a:clrScheme>";
  zip.file(part, xml
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${name}"`));
  // pptxgenjs writes crossesAt="autoZero", which PowerPoint repairs; the valid form is <c:crosses>
  for (const f of Object.keys(zip.files).filter((f) => /^ppt\/charts\/chart\d+\.xml$/.test(f))) {
    const c = await zip.file(f).async("string");
    zip.file(f, c.replace(/<c:crossesAt val="autoZero"\/>/g, '<c:crosses val="autoZero"/>'));
  }
  fs.writeFileSync(deckPath, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

module.exports = { applyTheme };
