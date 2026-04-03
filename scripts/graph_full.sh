#!/usr/bin/env bash
# Two-phase graph metrics on the full 32M-node corpus.
# Phase 1: PageRank + HITS (fits in memory)
# Phase 2: Leiden on giant component only (memory-managed)
#
# Usage: bash scripts/graph_full.sh

set -euo pipefail

VENV=".venv/bin/python3"
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "=== Phase 1: PageRank + HITS (full graph) ==="
echo "Start: $(date)"

$VENV -c "
import sys, logging, time, gc
sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
from scix.graph_metrics import load_graph, compute_pagerank, compute_hits, store_metrics
from scix.db import get_connection

logger = logging.getLogger('graph_full')

# Load graph
conn = get_connection()
graph, b2i, i2b = load_graph(conn)
n = graph.vcount()
logger.info(f'Graph: {n:,} nodes, {graph.ecount():,} edges')

# PageRank
pr = compute_pagerank(graph)

# HITS
hub, auth = compute_hits(graph)

# Store with NULL communities (Leiden runs in phase 2)
null_communities = [None] * n
stored = store_metrics(conn, pr, hub, auth, null_communities, null_communities, null_communities, i2b)
logger.info(f'Stored {stored:,} rows')
conn.close()

# Free memory before phase 2
del graph, b2i, i2b, pr, hub, auth, null_communities
gc.collect()
logger.info('Phase 1 complete, memory freed')
"

echo ""
echo "=== Phase 2: Leiden on giant component ==="
echo "Start: $(date)"

$VENV -c "
import sys, logging, time, gc
sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
from scix.graph_metrics import (
    load_graph, filter_isolated_nodes, extract_giant_component,
    compute_leiden, calibrate_resolution
)
from scix.db import get_connection
import psycopg, io

logger = logging.getLogger('graph_full')

conn = get_connection()

# Load graph
graph, b2i, i2b = load_graph(conn)
logger.info(f'Graph: {graph.vcount():,} nodes, {graph.ecount():,} edges')

# Filter isolated nodes
sub, sub_b2i, sub_i2b, isolated = filter_isolated_nodes(graph, b2i, i2b)
logger.info(f'Connected: {sub.vcount():,}, isolated: {len(isolated):,}')

# Free full graph
del graph, b2i, i2b, isolated
gc.collect()

# Extract giant component
giant, g_b2i, g_i2b, small = extract_giant_component(sub, sub_b2i, sub_i2b)
logger.info(f'Giant component: {giant.vcount():,} nodes, {giant.ecount():,} edges')
logger.info(f'Small components: {len(small):,} nodes')

# Free subgraph
del sub, sub_b2i, sub_i2b
gc.collect()

# Leiden at 3 resolutions on giant component
# Use fixed resolutions first (calibration is slow on large graphs)
for res_name, res_val in [('coarse', 0.001), ('medium', 0.01), ('fine', 0.1)]:
    logger.info(f'Running Leiden at {res_name} resolution ({res_val})...')
    membership = compute_leiden(giant, resolution=res_val, seed=42)
    n_communities = len(set(membership))
    logger.info(f'{res_name}: {n_communities} communities')

    # Store directly via SQL UPDATE (avoids re-creating full arrays)
    col = f'community_id_{res_name}'
    chunk_size = 50000
    updated = 0
    for i in range(0, len(membership), chunk_size):
        chunk_data = []
        for vid in range(i, min(i + chunk_size, len(membership))):
            bib = g_i2b[vid]
            cid = membership[vid]
            chunk_data.append((cid, bib))

        with conn.cursor() as cur:
            cur.executemany(
                f'UPDATE paper_metrics SET {col} = %s WHERE bibcode = %s',
                chunk_data,
            )
        conn.commit()
        updated += len(chunk_data)

    logger.info(f'Stored {updated:,} {res_name} community assignments')
    del membership
    gc.collect()

# Also assign small-component papers to community -1 (unassigned)
if small:
    small_list = list(small)
    chunk_size = 50000
    for i in range(0, len(small_list), chunk_size):
        chunk = small_list[i:i+chunk_size]
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE paper_metrics SET community_id_coarse = -1, community_id_medium = -1, community_id_fine = -1 WHERE bibcode = ANY(%s)',
                (chunk,),
            )
        conn.commit()
    logger.info(f'Assigned {len(small_list):,} small-component papers to community -1')

conn.close()
logger.info('Phase 2 complete')
"

echo ""
echo "=== Phase 3: Taxonomic communities ==="
echo "Start: $(date)"

$VENV -c "
import sys, logging
sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
import psycopg
from scix.db import get_connection

logger = logging.getLogger('graph_full')
conn = get_connection()

# Populate community_taxonomic from arxiv_class for all papers
# This extracts the first arxiv_class entry as the taxonomic community
with conn.cursor() as cur:
    cur.execute('''
        UPDATE paper_metrics pm
        SET community_taxonomic = p.arxiv_class[1]
        FROM papers p
        WHERE pm.bibcode = p.bibcode
          AND p.arxiv_class IS NOT NULL
          AND array_length(p.arxiv_class, 1) > 0
          AND pm.community_taxonomic IS NULL
    ''')
    updated = cur.rowcount
conn.commit()
logger.info(f'Updated {updated:,} papers with taxonomic community from arxiv_class')

# Also insert paper_metrics rows for papers that don't have them yet
with conn.cursor() as cur:
    cur.execute('''
        INSERT INTO paper_metrics (bibcode, community_taxonomic)
        SELECT p.bibcode, p.arxiv_class[1]
        FROM papers p
        LEFT JOIN paper_metrics pm ON pm.bibcode = p.bibcode
        WHERE pm.bibcode IS NULL
          AND p.arxiv_class IS NOT NULL
          AND array_length(p.arxiv_class, 1) > 0
        ON CONFLICT (bibcode) DO UPDATE SET
            community_taxonomic = EXCLUDED.community_taxonomic
    ''')
    inserted = cur.rowcount
conn.commit()
logger.info(f'Inserted {inserted:,} new paper_metrics rows with taxonomic community')

conn.close()
logger.info('Taxonomic communities complete')
"

echo ""
echo "=== All Done ==="
echo "End: $(date)"

# Final stats
$VENV -c "
import psycopg
conn = psycopg.connect('dbname=scix')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM paper_metrics WHERE pagerank IS NOT NULL')
pr = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM paper_metrics WHERE community_id_coarse IS NOT NULL')
cc = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM paper_metrics WHERE community_taxonomic IS NOT NULL')
ct = cur.fetchone()[0]
print(f'PageRank: {pr:,} papers')
print(f'Leiden communities: {cc:,} papers')
print(f'Taxonomic communities: {ct:,} papers')
conn.close()
"
