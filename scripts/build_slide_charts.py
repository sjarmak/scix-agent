"""Generate matplotlib charts for the slide deck."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("/home/ds/projects/scix_experiments/docs/slides_assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "xtick.color": "#222",
    "ytick.color": "#222",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

ACCENT = "#1f6feb"
ACCENT_DARK = "#0b4a9e"
MUTED = "#9ca3af"
GREEN = "#2aa876"
RED = "#d0564a"


def chart_edge_resolution():
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["6-year window\n(2021–2026, ~5M papers)", "Full corpus\n(1800–2026, 32.4M papers)"]
    values = [17.8, 99.6]
    colors = [RED, GREEN]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}%",
                ha="center", va="bottom", fontsize=18, fontweight="bold", color="#222")
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of citation edges resolved")
    ax.set_title("Corpus completeness determines graph validity",
                 fontsize=14, fontweight="bold", pad=14, color="#111")
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x)}%")
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.03,
             "82.2% of edges point to papers outside a 6-year window — graph analytics are structurally misleading",
             ha="center", fontsize=10, color="#555", style="italic")
    fig.savefig(OUT / "edge_resolution.png")
    plt.close(fig)


def chart_retrieval_eval():
    """Latest rebaseline results from results/retrieval_eval_50q_rebaseline.md"""
    methods = ["lexical", "hybrid_specter2", "specter2", "hybrid_indus", "indus", "nomic"]
    ndcg = [0.086, 0.321, 0.322, 0.333, 0.350, 0.374]
    colors = [MUTED, "#b6d4fe", "#86b6ff", "#5b95fc", ACCENT, ACCENT_DARK]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(methods, ndcg, color=colors, edgecolor="white")
    for bar, v in zip(bars, ndcg):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=11, color="#222")
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("nDCG@10")
    ax.set_title("50-query retrieval evaluation (citation-based ground truth)",
                 fontsize=14, fontweight="bold", pad=14, color="#111")
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, -0.04,
             "Surprise: general-purpose nomic beats domain-tuned INDUS. Evaluation methodology matters.",
             ha="center", fontsize=10, color="#555", style="italic")
    fig.savefig(OUT / "retrieval_eval.png")
    plt.close(fig)


def chart_corpus_stats():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    stats = [
        ("Papers", "32.4M", "1800–2026"),
        ("Citation edges", "299M", "99.6% resolved"),
        ("INDUS embeddings", "32.4M", "full coverage"),
    ]
    for ax, (label, value, sub) in zip(axes, stats):
        ax.axis("off")
        ax.text(0.5, 0.68, value, ha="center", va="center",
                fontsize=42, fontweight="bold", color=ACCENT_DARK)
        ax.text(0.5, 0.38, label, ha="center", va="center",
                fontsize=14, color="#222")
        ax.text(0.5, 0.22, sub, ha="center", va="center",
                fontsize=11, color="#666", style="italic")
    fig.suptitle("Corpus at a glance", fontsize=16, fontweight="bold", color="#111")
    fig.savefig(OUT / "corpus_stats.png")
    plt.close(fig)


def chart_architecture():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(x, y, w, h, label, sub="", color=ACCENT, alpha=1.0):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.02,rounding_size=0.12",
                                        linewidth=1.2, edgecolor=color,
                                        facecolor=color, alpha=alpha)
        ax.add_patch(rect)
        text_color = "white" if alpha > 0.5 else "#222"
        ax.text(x + w / 2, y + h / 2 + (0.12 if sub else 0), label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=text_color)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                    ha="center", va="center", fontsize=9, color=text_color, alpha=0.92)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.5))

    # Data layer
    box(0.2, 4.2, 2.3, 0.9, "NASA ADS", "32.4M papers, 299M edges", color="#0b4a9e")
    # Storage layer
    box(3.2, 4.2, 3.0, 0.9, "PostgreSQL 16 + pgvector", "HNSW, halfvec, BM25/tsvector", color=ACCENT)
    box(6.9, 4.2, 2.2, 0.9, "Embeddings", "INDUS (32.4M)", color=ACCENT)
    box(9.4, 4.2, 2.4, 0.9, "Graph metrics", "PageRank · Leiden ×3", color=ACCENT)
    # Retrieval layer
    box(1.8, 2.6, 2.6, 0.9, "Hybrid search", "BM25 + dense, RRF k=60", color="#2aa876")
    box(5.0, 2.6, 2.4, 0.9, "Graph tools", "citation_chain, find_gaps", color="#2aa876")
    box(8.0, 2.6, 2.6, 0.9, "Entity layer", "methods · datasets · instruments", color="#2aa876")
    # MCP
    box(3.8, 1.0, 4.4, 0.9, "MCP server  ·  13 tools",
        "search · read_paper · citation_graph · find_gaps · ...", color="#db8d3a")
    # Agent
    box(5.0, -0.2, 2.0, 0.7, "Agent", "Claude / research copilot", color="#222")

    # Arrows
    arrow(2.5, 4.65, 3.2, 4.65)
    arrow(6.2, 4.65, 6.9, 4.65)
    arrow(6.2, 4.65, 9.4, 4.65)
    arrow(3.1, 4.2, 3.1, 3.5)
    arrow(6.2, 4.2, 6.2, 3.5)
    arrow(9.3, 4.2, 9.3, 3.5)
    arrow(3.1, 2.6, 5.8, 1.9)
    arrow(6.2, 2.6, 6.2, 1.9)
    arrow(9.3, 2.6, 6.6, 1.9)
    arrow(6.0, 1.0, 6.0, 0.5)

    ax.text(6, 5.25, "Architecture: single-box Postgres powers an agent-native knowledge layer",
            ha="center", fontsize=14, fontweight="bold", color="#111")
    fig.savefig(OUT / "architecture.png")
    plt.close(fig)


def chart_tool_count():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    categories = ["Search", "Paper", "Graph", "Entity", "Session"]
    old = [4, 3, 6, 2, 7]  # 22
    new = [2, 2, 4, 2, 3]  # 13
    x = np.arange(len(categories))
    w = 0.38
    ax.bar(x - w/2, old, w, label="v1 (22 tools)", color=MUTED, edgecolor="white")
    ax.bar(x + w/2, new, w, label="v2 (13 tools)", color=ACCENT, edgecolor="white")
    for i, (o, n) in enumerate(zip(old, new)):
        ax.text(i - w/2, o + 0.1, str(o), ha="center", fontsize=10, color="#555")
        ax.text(i + w/2, n + 0.1, str(n), ha="center", fontsize=10, fontweight="bold", color=ACCENT_DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("# MCP tools")
    ax.set_title("Tool count discipline: 22 → 13", fontsize=14, fontweight="bold", pad=14, color="#111")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.03,
             "Premortem finding: >15 tools degrades agent selection accuracy",
             ha="center", fontsize=10, color="#555", style="italic")
    fig.savefig(OUT / "tool_count.png")
    plt.close(fig)


def chart_embedding_pipeline():
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stages = ["Naive loop", "+ multiprocess", "+ binary COPY", "+ drop indexes\nduring load"]
    tput = [32, 180, 380, 508]
    bars = ax.bar(stages, tput, color=[MUTED, "#86b6ff", ACCENT, ACCENT_DARK], edgecolor="white")
    for bar, v in zip(bars, tput):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 12, f"{v}", ha="center",
                fontsize=12, fontweight="bold", color="#222")
    ax.set_ylabel("Embedding throughput (records/sec)")
    ax.set_title("32 → 508 rec/s: embedding pipeline optimization",
                 fontsize=14, fontweight="bold", pad=14, color="#111")
    ax.set_ylim(0, 600)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, -0.03,
             "RTX 5090, INDUS (768d), binary COPY writer + UNLOGGED bulk load + dropped HNSW",
             ha="center", fontsize=10, color="#555", style="italic")
    fig.savefig(OUT / "embedding_pipeline.png")
    plt.close(fig)


if __name__ == "__main__":
    chart_edge_resolution()
    chart_retrieval_eval()
    chart_corpus_stats()
    chart_architecture()
    chart_tool_count()
    chart_embedding_pipeline()
    print("Generated charts in", OUT)
