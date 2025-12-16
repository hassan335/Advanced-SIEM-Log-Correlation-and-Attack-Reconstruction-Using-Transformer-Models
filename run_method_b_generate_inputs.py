import numpy as np
import pandas as pd
from pathlib import Path
import torch

from transformers import BertForSequenceClassification, BertConfig

from recon_method_b_attention import (
    build_edges_attention_method_b,
    merge_edges_across_windows,
    prune_edges,
    extract_max_weight_path_dag,
    plot_graph,
    aggregate_attention,
    plot_attention_heatmap
)


# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/model2_classifier_finetuned")   # <-- your saved folder
OUT_DIR = Path("results/methodB_attention")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Special tokens 
PAD_ID  = 0
CLS_ID  = 1
SEP_ID  = 2
MASK_ID = 3
EVENT_OFFSET = 4

SEQ_LEN = 50
MAX_LEN = SEQ_LEN + 2  # [CLS] + events + [SEP]

# how many windows to process 
MAX_WINDOWS = 300      


# Helper: build BERT input ids from window tokens

def build_input_ids(window_tokens: np.ndarray):
    # window_tokens: shape (SEQ_LEN,) values 0..K-1
    seq = window_tokens.astype(np.int64) + EVENT_OFFSET  # events -> vocab range
    input_ids = np.zeros((MAX_LEN,), dtype=np.int64)
    input_ids[0] = CLS_ID
    input_ids[1:1+SEQ_LEN] = seq
    input_ids[1+SEQ_LEN] = SEP_ID
    attention_mask = (input_ids != PAD_ID).astype(np.int64)
    return input_ids, attention_mask

# Load token windows

X_test_tok = np.load(DATA_DIR / "X_test_tok.npy")  # (N, SEQ_LEN)
N = min(len(X_test_tok), MAX_WINDOWS)
X = X_test_tok[:N]

# infer vocab size from tokens (K)
K = int(X_test_tok.max()) + 1
VOCAB_SIZE = EVENT_OFFSET + K

print("DEVICE:", DEVICE)
print("Loaded windows:", N)
print("K:", K, "VOCAB_SIZE:", VOCAB_SIZE)


# Load fine-tuned classifier model


cls_model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
cls_model.eval()

# Build synthetic events + seq_windows + window_attn

events_rows = []
seq_windows = []
window_attn = {}

window_scores = []

event_global_id = 0

with torch.no_grad():
    for w_id in range(N):
        input_ids_np, attn_np = build_input_ids(X[w_id])

        input_ids = torch.tensor(input_ids_np, dtype=torch.long).unsqueeze(0).to(DEVICE)
        attention_mask = torch.tensor(attn_np, dtype=torch.long).unsqueeze(0).to(DEVICE)

        out = cls_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )

        probs = torch.softmax(out.logits, dim=1)[0].detach().cpu().numpy()
        # fast window-level anomaly score: 1 - max confidence
        win_anom = float(1.0 - probs.max())
        window_scores.append(win_anom)

        # attentions is tuple(layers), each (B, heads, T, T)
        att = out.attentions  # tuple
        # stack -> (layers, heads, T, T)
        att_stack = np.stack([a[0].detach().cpu().numpy() for a in att], axis=0)

      
        # (layers, heads, SEQ_LEN, SEQ_LEN)
        att_events = att_stack[:, :, 1:-1, 1:-1]
        window_attn[w_id] = att_events

        # build global event ids for this window (unique ids)
        window_event_ids = []
        for pos in range(SEQ_LEN):
            eid = event_global_id
            event_global_id += 1
            window_event_ids.append(eid)

            events_rows.append({
                "event_id": eid,
                "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=eid),
                "host": f"host_{w_id % 5}",        # synthetic host
                "user": f"user_{w_id % 3}",        # synthetic user
                "event_type": f"cluster_{int(X[w_id, pos])}",  # token cluster as type
                "anomaly_score": win_anom
            })

        seq_windows.append(window_event_ids)

# events dataframe
events = pd.DataFrame(events_rows)


# Define anomalous events using top 5% anomaly_score

thr = events["anomaly_score"].quantile(0.95)
events["is_anom"] = events["anomaly_score"] >= thr
print("Anomaly threshold (top 5%):", thr)
print("Anomalous events:", int(events["is_anom"].sum()), "/", len(events))


# RUN METHOD B (Attention-Based Linkage)

edges_raw = build_edges_attention_method_b(
    events=events,
    seq_windows=seq_windows,
    window_attn=window_attn,
    threshold=0.05,  
    top_k=5,         
    agg_mode="sum",
    last_n_layers=None
)

edges_merged = merge_edges_across_windows(edges_raw)
edges_pruned = prune_edges(edges_merged, top_k_per_dst=3, min_weight_sum=None)

best_path = extract_max_weight_path_dag(events, edges_pruned, weight_col="weight_sum")
print("Best reconstructed chain event_ids:", best_path[:30])

# Save CSV outputs
events.to_csv(OUT_DIR / "events_generated.csv", index=False)
edges_raw.to_csv(OUT_DIR / "edges_raw.csv", index=False)
edges_merged.to_csv(OUT_DIR / "edges_merged.csv", index=False)
edges_pruned.to_csv(OUT_DIR / "edges_pruned.csv", index=False)

# Graph plot
plot_graph(edges_pruned, events, best_path, out_png=str(OUT_DIR / "methodB_graph.png"))

# One attention heatmap example (window 0)
att_agg = aggregate_attention(window_attn[0], mode="sum")
plot_attention_heatmap(att_agg, seq_windows[0], out_png=str(OUT_DIR / "methodB_attention_heatmap.png"))

# Print a readable timeline for the best path
if best_path:
    timeline = events.set_index("event_id").loc[best_path][
        ["timestamp", "host", "user", "event_type", "anomaly_score", "is_anom"]
    ].reset_index()
    timeline.to_csv(OUT_DIR / "best_chain_timeline.csv", index=False)
    print("\nSaved best chain timeline to:", (OUT_DIR / "best_chain_timeline.csv").resolve())

print("\nSaved all outputs to:", OUT_DIR.resolve())
