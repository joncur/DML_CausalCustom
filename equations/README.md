# Portable equations

These SVG files display the equations in the repository's Markdown documents
without requiring client-side MathJax. White backgrounds keep the formulas
legible in both light and dark views. All editable LaTeX is in `source.json`;
display equations also have expandable copyable source beside their images.

To regenerate after editing `source.json`, install the development-only
`mathjax-full` dependency and run `node render_equations.cjs` from the repository
root. This dependency is not used by the causal models or pipelines.
