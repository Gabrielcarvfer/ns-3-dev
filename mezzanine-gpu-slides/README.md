# Mezzanine GPU slides

Beamer deck documenting the WebGPU "mezzanine" GPU offload of the ns-3
contrib/nr 3GPP TR 38.901 channel model.

## Build

```
latexmk -pdf main.tex
```

or, without latexmk:

```
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # run twice for the TOC / refs
```

Output: `main.pdf` (16:9, ~36 pages including metropolis section pages).

## Required LaTeX packages

- `beamer`
- `metropolis` (beamer theme; the preamble falls back to `Madrid`+`seahorse`
  automatically if `beamerthememetropolis.sty` is not found)
- `tikz` (libraries: arrows.meta, positioning, shapes.geometric, fit,
  backgrounds, calc)
- `pgfplots` (the two speedup bar charts)
- `amsmath`, `amssymb`, `booktabs`

On TeX Live / MiKTeX these are in the standard distribution. For metropolis:
`tlmgr install beamertheme-metropolis pgfopts`.

## Notes

- All architecture/flow diagrams are TikZ; the speedup bars are pgfplots.
- Numbers trace to the project memory notes and `GPU_PERF_OPTIMIZATIONS.md` /
  `src/spectrum/doc/spectrum.rst`. See the final report for the few figures
  that should be re-checked.
