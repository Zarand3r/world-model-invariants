Clean source archive for arXiv. Compiles standalone with `tectonic -X compile main.tex`
(uses `main.bbl` directly, as arXiv does, rather than re-running BibTeX).

Contains only `main.tex`, `main.bbl`, the eight section files, and the five figures actually
included. No `.aux`, `.log`, `.blg`, backup files, or unused figures. All filenames are
arXiv-safe. Figures embed CID TrueType (fonttype 42); no Type 3 fonts anywhere.

Suggested metadata: primary `cs.LG`, cross-list `cs.AI`.
Comments field: "9 pages, 5 figures. Code at https://github.com/Zarand3r/latent-noether-paper1"
