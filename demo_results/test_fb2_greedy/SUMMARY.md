# test.fb2 greedy demo

This folder contains the small demo FB2 and the existing processed result snapshot for manual review.

Important: the demo analysis was not re-run while preparing this package.

## Files

- `test_books/demo/test.fb2` - small FB2 demo book committed to git.
- `thought_chain_analysis_report.md` - human-readable processed result snapshot.
- `thought_chain_analysis_report.json` - structured processed result snapshot.
- `quality_report.md` / `quality_report.json` - compact quality summary.

## Result summary

- book_id: 18
- title: Текст на 50 предложений
- status: ready
- sentences: 50
- meaningful_thoughts: 50
- sequential_groups: 9
- global_blocks: 9
- memberships: 50
- fallback_thoughts: 0
- invalid_json_thoughts: 0

## Run modes now available

- Default production mode: `greedy`.
- Demonstration/strict mode: `strict pairwise` via `-StrictPairwise`.
