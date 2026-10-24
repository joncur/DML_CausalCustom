// Development-only equation renderer. npm install --no-save mathjax-full@3.2.2
// Run: node render_equations.cjs from the repository directory.
const fs = require('node:fs');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', {InputJax:new TeX({packages:AllPackages}),
                                 OutputJax:new SVG({fontCache:'none'})});
const source = JSON.parse(fs.readFileSync('equations/source.json','utf8'));
for (const entry of source) {
  const html = adaptor.outerHTML(doc.convert(entry.latex,{display:entry.display}));
  if (html.includes('data-mjx-error')) throw new Error('Math error in '+entry.file);
  let svg = html.slice(html.indexOf('<svg'),html.lastIndexOf('</svg>')+6);
  if (!svg.includes('xmlns=')) svg=svg.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ');
  svg=svg.replace(/currentColor/g,'#111111');
  // Explicit dimensions and white backing stay legible without MathJax, including dark mode.
  svg=svg.replace(/(width|height)="([0-9.]+)ex"/g,(_,key,v)=>`${key}="${Math.ceil(Number(v)*8.5)}px"`);
  const box=svg.match(/viewBox="([^"]+)"/)[1].split(' ').map(Number);
  const rectangle=`<rect x="${box[0]}" y="${box[1]}" width="${box[2]}" height="${box[3]}" fill="white"/>`;
  svg=svg.replace(/(<svg[^>]*>)/,'$1'+rectangle);
  fs.writeFileSync('equations/'+entry.file,svg+'\n');
}
console.log(`Rendered ${source.length} vector equations.`);
