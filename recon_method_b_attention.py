import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception:
    nx = None


def aggregate_attention(att: np.ndarray, mode: str = "sum", last_n_layers: int | None = None) -> np.ndarray:
   
    if att.ndim != 4:
        raise ValueError(f"Expected attention shape (layers, heads, L, L). Got {att.shape}")

    if last_n_layers is not None and last_n_layers > 0:
        att = att[-last_n_layers:]  # keep last N layers

    if mode == "sum":
        att_agg = att.sum(axis=(0, 1))
    elif mode == "mean":
        att_agg = att.mean(axis=(0, 1))
    else:
        raise ValueError("mode must be 'sum' or 'mean'")

    return att_agg


def build_edges_attention_method_b(
    events: pd.DataFrame,
    seq_windows: list[list[int]],
    window_attn: dict[int, np.ndarray],
    threshold: float | None = 0.05,
    top_k: int = 5,
    agg_mode: str = "sum",
    last_n_layers: int | None = None,
) -> pd.DataFrame:
   
    if "event_id" not in events.columns:
        raise ValueError("events must contain column 'event_id'")
    if "is_anom" not in events.columns:
        raise ValueError("events must contain boolean column 'is_anom'")

    e = events.set_index("event_id", drop=False)
    edges = []

    for w_id, window in enumerate(seq_windows):
        if w_id not in window_attn:
            continue

        att = window_attn[w_id]  # (layers, heads, L, L)
        L = len(window)
        if att.shape[-1] != L:
            # mismatch → skip to avoid wrong mapping
            continue

        att_agg = aggregate_attention(att, mode=agg_mode, last_n_layers=last_n_layers)  # (L,L)

        # Find anomalous positions in this window
        anom_pos = []
        for j in range(L):
            eid_j = int(window[j])
            if eid_j in e.index and bool(e.loc[eid_j, "is_anom"]):
                anom_pos.append(j)

        for j in anom_pos:
            # only prior tokens i < j
            if j <= 0:
                continue

            col = att_agg[:j, j].astype(float)  # (j,)
            if col.size == 0:
                continue

            # take top_k influencers
            idx_sorted = np.argsort(col)[::-1]
            idx_top = idx_sorted[:top_k]

            for i in idx_top:
                w = float(col[i])

                if threshold is not None and w <= threshold:
                    continue

                src_eid = int(window[i])
                dst_eid = int(window[j])

                # skip if ids not in events
                if src_eid not in e.index or dst_eid not in e.index:
                    continue

                edges.append({
                    "window_id": w_id,
                    "src_event_id": src_eid,
                    "dst_event_id": dst_eid,
                    "weight": w
                })

    return pd.DataFrame(edges)


def merge_edges_across_windows(edges: pd.DataFrame) -> pd.DataFrame:
   
    if edges.empty:
        return edges

    g = edges.groupby(["src_event_id", "dst_event_id"], as_index=False).agg(
        weight_sum=("weight", "sum"),
        weight_mean=("weight", "mean"),
        support=("weight", "count"),
    )
    return g.sort_values(["weight_sum", "support"], ascending=False).reset_index(drop=True)


def prune_edges(edges_merged: pd.DataFrame, top_k_per_dst: int = 3, min_weight_sum: float | None = None) -> pd.DataFrame:
   
    if edges_merged.empty:
        return edges_merged

    df = edges_merged.copy()
    if min_weight_sum is not None:
        df = df[df["weight_sum"] >= float(min_weight_sum)]

    # top-k incoming per dst
    df = df.sort_values("weight_sum", ascending=False)
    df["rank_in"] = df.groupby("dst_event_id").cumcount() + 1
    df = df[df["rank_in"] <= top_k_per_dst].drop(columns=["rank_in"])
    return df.reset_index(drop=True)


def extract_max_weight_path_dag(
    events: pd.DataFrame,
    edges: pd.DataFrame,
    weight_col: str = "weight_sum"
) -> list[int]:
   
    if edges.empty:
        return []

    e = events.copy()
    if "timestamp" in e.columns:
        e["timestamp"] = pd.to_datetime(e["timestamp"], errors="coerce")
        e = e.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    else:
        e = e.sort_values(["event_id"]).reset_index(drop=True)

    order = e["event_id"].tolist()
    pos = {int(eid): idx for idx, eid in enumerate(order)}

    # Build adjacency
    incoming = {int(eid): [] for eid in order}
    for _, r in edges.iterrows():
        u = int(r["src_event_id"])
        v = int(r["dst_event_id"])
        if u in pos and v in pos and pos[u] < pos[v]:  # enforce DAG direction
            incoming[v].append((u, float(r[weight_col])))

    # DP: best score to reach node, and backpointer
    best = {int(eid): 0.0 for eid in order}
    prev = {int(eid): None for eid in order}

    for v in order:
        best_v = best[v]
        for u, w in incoming[v]:
            cand = best[u] + w
            if cand > best_v:
                best_v = cand
                prev[v] = u
        best[v] = best_v

    # end node with max score
    end = max(order, key=lambda eid: best[int(eid)])
    if best[int(end)] <= 0:
        return [int(end)]

    # backtrack
    path = []
    cur = int(end)
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def plot_graph(edges: pd.DataFrame, events: pd.DataFrame, highlight_path: list[int], out_png: str):
   
    if nx is None:
        print("networkx not installed; skipping graph plot.")
        return
    if edges.empty:
        print("No edges to plot.")
        return

    e = events.set_index("event_id", drop=False)

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        u = int(r["src_event_id"])
        v = int(r["dst_event_id"])
        w = float(r["weight_sum"])
        G.add_edge(u, v, weight=w)

    # labels
    def node_label(eid: int) -> str:
        if eid not in e.index:
            return str(eid)
        et = e.loc[eid, "event_type"] if "event_type" in e.columns else ""
        hs = e.loc[eid, "host"] if "host" in e.columns else ""
        return f"{eid}\n{hs}\n{et}".strip()

    labels = {n: node_label(n) for n in G.nodes()}

    # layout
    pos = nx.spring_layout(G, seed=42)

    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    # draw edges (default)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.35)

    # highlight max path
    if len(highlight_path) >= 2:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, arrows=True, width=2.5)

    plt.title("Method B: Attention-Based Event Linkage Graph")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_attention_heatmap(att_agg: np.ndarray, window_event_ids: list[int], out_png: str):
   
    plt.figure(figsize=(8, 7))
    plt.imshow(att_agg, interpolation="nearest")
    plt.colorbar()
    plt.title("Aggregated Attention Heatmap (Method B)")
    plt.xlabel("Influenced event position (j)")
    plt.ylabel("Influencer event position (i)")
    # optional tick labels (can be heavy if L large)
    if len(window_event_ids) <= 30:
        plt.xticks(range(len(window_event_ids)), window_event_ids, rotation=90, fontsize=7)
        plt.yticks(range(len(window_event_ids)), window_event_ids, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
