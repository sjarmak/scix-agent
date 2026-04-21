#!/usr/bin/env bash
#
# Drive a scripted sequence of agent-trace events into the running viz
# server. Each event lands on the SSE stream that agent_trace.html subscribes
# to, flashing the touched papers on the UMAP overlay and narrating the tool
# call in the right-hand panel.
#
# The bibcodes below are picked from `paper_umap_2d` so the flashes land on
# real points in the current projection. Picking other bibcodes that aren't
# in the 100K-paper sample will still narrate but won't highlight on the map.
#
# Usage:
#   scripts/viz/demo_scenario.sh            # runs all three scenarios
#   scripts/viz/demo_scenario.sh survey     # only the literature-survey one
#   scripts/viz/demo_scenario.sh --host URL # point at a non-local viz server

set -euo pipefail

HOST="http://127.0.0.1:8765"
SCENARIO="${1:-all}"
if [[ "${1:-}" == --host ]]; then
  HOST="${2:?--host requires a URL}"
  SCENARIO="${3:-all}"
fi

post() {
  local tool="$1"
  local latency="$2"
  shift 2
  local bibs_json="[]"
  if [ "$#" -gt 0 ]; then
    bibs_json="["
    local first=1
    for b in "$@"; do
      [ $first -eq 0 ] && bibs_json+=","
      bibs_json+="\"$b\""
      first=0
    done
    bibs_json+="]"
  fi
  curl -sfS -X POST -H "Content-Type: application/json" \
    -d "{\"tool_name\":\"$tool\",\"latency_ms\":$latency,\"bibcodes\":$bibs_json}" \
    "$HOST/viz/api/trace/publish" >/dev/null
  echo "  -> $tool  ($#)  lat=${latency}ms"
  sleep 0.8
}

scenario_survey() {
  echo "[scenario: literature-survey] galaxy clusters"
  post search 82 "2023APS..MART15006K" "2022MNRAS.513.5681M" "2021ApJ...915...81A"
  post concept_search 41
  post citation_chain 120 "2022MNRAS.513.5681M" "2020ApJ...888...85A"
  post get_paper 18 "2020ApJ...888...85A"
  post read_paper 340 "2020ApJ...888...85A"
}

scenario_methods() {
  echo "[scenario: related-methods] deep learning for redshifts"
  post search 64 "2023APS..MART15006K"
  post citation_similarity 95 "2023APS..MART15006K" "2021ApJ...915...81A"
  post graph_context 210 "2023APS..MART15006K" "2022MNRAS.513.5681M" "2021ApJ...915...81A"
  post get_paper 22 "2021ApJ...915...81A"
}

scenario_disambig() {
  echo "[scenario: entity-disambiguation]"
  post entity 30
  post entity_context 48 "2023APS..MART15006K"
  post find_gaps 170
  post temporal_evolution 88
}

case "$SCENARIO" in
  all|"")
    scenario_survey;   sleep 1.2
    scenario_methods;  sleep 1.2
    scenario_disambig
    ;;
  survey) scenario_survey ;;
  methods) scenario_methods ;;
  disambig) scenario_disambig ;;
  *)
    echo "unknown scenario: $SCENARIO" >&2
    echo "usage: $0 [all|survey|methods|disambig]" >&2
    exit 2
    ;;
esac

echo "done"
