#!/usr/bin/env bash
# Rebuild paper/arxiv-submission.tar.gz from the repo. Never hand-copy files into the archive:
# doing that once left two stale figures in it that differed from the paper's own.
set -euo pipefail
# Which manuscript to package. Defaults to `paper` for backwards compatibility; `paper1.2` is the
# current one. Hardcoding the directory meant the archive silently kept targeting the superseded
# manuscript after the fork.
DIR="${1:-paper}"
cd "$(dirname "$0")/../$DIR"
OUT=arxiv-submission
rm -rf "$OUT" && mkdir -p "$OUT/sections" "$OUT/figures"

# main.bbl must be shipped, and refs.bib alongside it: a build from a clean extraction re-runs
# BibTeX, and without the database every citation renders as [?].
tectonic -X compile main.tex --keep-intermediates >/dev/null 2>&1

cp main.tex refs.bib main.bbl "$OUT/"
# The venue style files are not in TeX Live; a clean extraction cannot build without them.
for f in *.sty *.bst math_commands.tex; do [ -e "$f" ] && cp "$f" "$OUT/"; done
cp sections/*.tex "$OUT/sections/"
# only the figures the paper actually includes
for f in $(grep -oh 'figures/[a-z0-9_]*\.pdf' sections/*.tex | sort -u); do cp "$f" "$OUT/figures/"; done

rm -f arxiv-submission.tar.gz
( cd "$OUT" && tar -czf ../arxiv-submission.tar.gz . )
echo "wrote $DIR/arxiv-submission.tar.gz"
tar -tzf arxiv-submission.tar.gz | grep -v '/$' | sort | sed 's/^/  /'
