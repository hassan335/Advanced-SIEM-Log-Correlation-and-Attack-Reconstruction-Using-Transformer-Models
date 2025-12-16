import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import BertForMaskedLM

# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENTS_CSV = Path("results/methodB_attention/events_generated.csv")
X_TOK_PATH = Path("data/processed/X_test_tok.npy")
MLM_DIR = Path("models/model2_mlm_pretrained")

OUT_DIR = Path("results/methodC_greedy")
OUT_DIR.mkdir(parents=True, exist_ok=True)


PAD_ID  = 0
CLS_ID  = 1
SEP_ID  = 2
MASK_ID = 3
EVENT_OFFSET = 4

SEQ_LEN = 50
MAX_LEN = SEQ_LEN + 2

LOOKAHEAD = 20    
P_MIN = 0.20      
MAX_CHAIN_LEN = 20

# how many seeds to try (CPU safe)
MAX_SEEDS = 50


def build_input_ids(window_tokens: np.ndarray):
    """window_tokens: (SEQ_LEN,) in [0..K-1]"""
    seq = window_tokens.astype(np.int64) + EVENT_OFFSET
    input_ids = np.zeros((MAX_LEN,), dtype=np.int64)
    input_ids[0] = CLS_ID
    input_ids[1:1+SEQ_LEN] = seq
    input_ids[1+SEQ_LEN] = SEP_ID
    attn = (input_ids != PAD_ID).astype(np.int64)
    return input_ids, attn


@torch.no_grad()
def prob_of_token_at_pos(mlm_model, window_tokens, pos, token_id, K):
    
    input_ids_np, attn_np = build_input_ids(window_tokens)

    # BERT position of event pos is 1+pos 
    bert_pos = 1 + pos
    masked_ids = input_ids_np.copy()
    masked_ids[bert_pos] = MASK_ID

    input_ids = torch.tensor(masked_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    attn = torch.tensor(attn_np, dtype=torch.long).unsqueeze(0).to(DEVICE)

    out = mlm_model(input_ids=input_ids, attention_mask=attn, return_dict=True)
    logits = out.logits[0, bert_pos]  # (VOCAB,)

    start = EVENT_OFFSET
    end = EVENT_OFFSET + K
    probs = torch.softmax(logits[start:end], dim=0)  # (K,)
    return float(probs[int(token_id)].item())


def greedy_chain_for_seed(events, X_tok, mlm_model, seed_event_id, K):
    
    seed_event_id = int(seed_event_id)

  
    w_id = seed_event_id // SEQ_LEN
    pos = seed_event_id % SEQ_LEN

    if w_id < 0 or w_id >= len(X_tok):
        return None

    window_tokens = X_tok[w_id]  # (SEQ_LEN,)
    chain = [seed_event_id]
    probs_used = []
    cur_pos = pos

    # Greedy steps
    for _ in range(MAX_CHAIN_LEN - 1):
        # candidate positions are observed future events
        cand_positions = list(range(cur_pos + 1, min(cur_pos + 1 + LOOKAHEAD, SEQ_LEN)))
        if not cand_positions:
            break

        cand_event_ids = [w_id * SEQ_LEN + p for p in cand_positions]
        cand_token_ids = [int(window_tokens[p]) for p in cand_positions]

        # compute P(candidate | context) for each candidate by masking that candidate position
        cand_probs = []
        for p, tok in zip(cand_positions, cand_token_ids):
            pr = prob_of_token_at_pos(mlm_model, window_tokens, p, tok, K)
            cand_probs.append(pr)

        best_idx = int(np.argmax(cand_probs))
        best_prob = float(cand_probs[best_idx])

        if best_prob < P_MIN:
            break

        next_event_id = int(cand_event_ids[best_idx])
        chain.append(next_event_id)
        probs_used.append(best_prob)
        cur_pos = int(cand_positions[best_idx])

    # chain score (sum log probs)
    if probs_used:
        score = float(np.sum(np.log(np.maximum(probs_used, 1e-12))))
    else:
        score = float("-inf")

    return {
        "seed_event_id": seed_event_id,
        "window_id": int(w_id),
        "start_pos": int(pos),
        "chain": chain,
        "probs": probs_used,
        "score_sum_logprob": score,
        "avg_prob": float(np.mean(probs_used)) if probs_used else 0.0,
        "length": len(chain),
    }


if __name__ == "__main__":
    # Load inputs
    events = pd.read_csv(EVENTS_CSV)
    X_tok = np.load(X_TOK_PATH)  # (num_windows, 50)
    K = int(X_tok.max()) + 1

    mlm_model = BertForMaskedLM.from_pretrained(MLM_DIR).to(DEVICE)
    mlm_model.eval()

    print("DEVICE:", DEVICE)
    print("Loaded events:", len(events))
    print("Windows:", len(X_tok), "SEQ_LEN:", SEQ_LEN)
    print("K:", K)

    # choose anomalous seeds
    if "is_anom" not in events.columns:
        raise ValueError("events_generated.csv must contain is_anom column. (It should from Method B script.)")

    seeds = events.loc[events["is_anom"] == True, "event_id"].astype(int).tolist()
    seeds = seeds[:MAX_SEEDS]
    print("Using seeds:", len(seeds))

    results = []
    for s in seeds:
        r = greedy_chain_for_seed(events, X_tok, mlm_model, s, K)
        if r is not None and r["length"] >= 2:
            results.append(r)

    if not results:
        print("No chains found (try lowering P_MIN or increasing LOOKAHEAD).")
        exit(0)

    # rank by score
    results = sorted(results, key=lambda x: x["score_sum_logprob"], reverse=True)

    # save summary
    summary = pd.DataFrame([{
        "seed_event_id": r["seed_event_id"],
        "window_id": r["window_id"],
        "length": r["length"],
        "avg_prob": r["avg_prob"],
        "score_sum_logprob": r["score_sum_logprob"],
        "chain": " -> ".join(map(str, r["chain"]))
    } for r in results])

    summary.to_csv(OUT_DIR / "methodC_chains_ranked.csv", index=False)
    print("Saved ranked chains to:", (OUT_DIR / "methodC_chains_ranked.csv").resolve())

    # save one example chain timeline (best chain)
    best = results[0]
    chain_ids = best["chain"]

    # build timeline rows from events table
    ev = events.set_index("event_id")
    timeline = ev.loc[chain_ids].reset_index()

    # attach step + probability used (prob for transition into this step)
    step_probs = [None] + best["probs"]
    timeline.insert(0, "step", list(range(1, len(chain_ids) + 1)))
    timeline["transition_prob_from_prev"] = step_probs

    timeline.to_csv(OUT_DIR / "methodC_best_chain_timeline.csv", index=False)
    print("Saved best chain timeline to:", (OUT_DIR / "methodC_best_chain_timeline.csv").resolve())

    print("\nBest chain:", " -> ".join(map(str, chain_ids)))
    print("Avg prob:", best["avg_prob"], "Score(sum log prob):", best["score_sum_logprob"])
