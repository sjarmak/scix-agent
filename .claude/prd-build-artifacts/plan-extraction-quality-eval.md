# Plan: extraction-quality-eval

## Step 1: Create scripts/eval_extraction_quality.py

### Core Functions

1. `sample_papers(conn, n=50)` — Query n random bibcodes from extractions table (SELECT DISTINCT bibcode ... ORDER BY RANDOM() LIMIT n). Return list of bibcodes.

2. `get_mentions(conn, bibcodes)` — Query all extraction rows for given bibcodes. Parse payload->"entities" arrays. Return dict mapping bibcode to list of mention strings (flattened across extraction_types).

3. `evaluate_mentions(conn, mentions_by_bibcode, fuzzy=False)` — For each mention, run EntityResolver.resolve(). Collect:
   - total_mentions count
   - resolved_mentions count (got >= 1 candidate)
   - unmatched_mentions list (got 0 candidates)
   - match_method_counts dict (count per match_method from top candidate)
   - resolution_rate = resolved / total (proxy for precision)
   - recall_proxy = resolved / total (same metric since no ground truth)
   - Return a frozen dataclass EvalResult with all these fields.

4. `format_report(result: EvalResult)` — Format markdown report string with:
   - Summary stats (papers sampled, total mentions, resolution rate)
   - Match method distribution table
   - Unmatched mention examples (up to 20)

5. `write_report(report_str, output_path)` — Write to file.

6. `main()` — argparse CLI: --dsn, --sample-size (default 50), --fuzzy, --output, -v. Wire everything together.

### Data Flow

extractions table -> sample bibcodes -> get mentions -> resolve each -> compute stats -> write report

## Step 2: Create tests/test_eval_extraction_quality.py

Tests with mocked DB and EntityResolver:

- test_sample_papers_queries_extractions
- test_get_mentions_parses_payload
- test_evaluate_mentions_computes_stats
- test_evaluate_mentions_tracks_unmatched
- test_report_includes_match_distribution
- test_report_includes_unmatched_examples
- test_script_importable
