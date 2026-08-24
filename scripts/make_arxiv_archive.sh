#!/usr/bin/env bash
# Rebuild paper/arxiv-submission.tar.gz from the repo. Never hand-copy files into the archive:
# doing that once left two stale figures in it that differed from the paper's own.
set -euo pipefail
cd "$(dirname "$0")/../paper"
OUT=arxiv-submission
rm -rf "$OUT" && mkdir -p "$OUT/sections" "$OUT/figures"

# main.bbl must be shipped, and refs.bib alongside it: a build from a clean extraction re-runs
# BibTeX, and without the database every citation renders as [?].
tectonic -X compile main.tex --keep-intermediates >/dev/null 2>&1

cp main.tex refs.bib main.bbl "$OUT/"
cp sections/*.tex "$OUT/sections/"
# only the figures the paper actually includes
for f in $(grep -oh 'figures/[a-z0-9_]*\.pdf' sections/*.tex | sort -u); do cp "$f" "$OUT/figures/"; done

rm -f arxiv-submission.tar.gz
( cd "$OUT" && tar -czf ../arxiv-submission.tar.gz . )
echo "wrote paper/arxiv-submission.tar.gz"
tar -tzf arxiv-submission.tar.gz | grep -v '/$' | sort | sed 's/^/  /'
