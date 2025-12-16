import os
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification



# Config

DATA_DIR = Path("data/processed")
OUT_DIR = Path("results/model2_bert_log_transformer")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Token IDs (custom vocab)
PAD_ID  = 0
CLS_ID  = 1
SEP_ID  = 2
MASK_ID = 3
EVENT_OFFSET = 4  # event tokens start from 4

# Your sequence length (from Model 1)
SEQ_LEN = 50
MAX_LEN = SEQ_LEN + 2  # [CLS] + events + [SEP]

# Training
BATCH_SIZE = 128
LR_PRETRAIN = 2e-4
LR_FINETUNE = 2e-4
EPOCHS_PRETRAIN = 3
EPOCHS_FINETUNE = 5

MLM_PROB = 0.15

# BERT size 
HIDDEN = 128
LAYERS = 4
HEADS  = 4
DROPOUT = 0.1


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# Helpers: build inputs

def add_special_tokens_and_pad(seq_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    # shift event ids into vocab range
    seq = seq_tokens.astype(np.int64) + EVENT_OFFSET  # events -> [4..]
    # truncate or pad to SEQ_LEN 
    seq = seq[:SEQ_LEN]
    if len(seq) < SEQ_LEN:
        seq = np.pad(seq, (0, SEQ_LEN - len(seq)), constant_values=PAD_ID)

    input_ids = np.zeros((MAX_LEN,), dtype=np.int64)
    input_ids[0] = CLS_ID
    input_ids[1:1+SEQ_LEN] = seq
    input_ids[1+SEQ_LEN] = SEP_ID

    
    attn = (input_ids != PAD_ID).astype(np.int64)
    return input_ids, attn


def make_mlm_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor, mlm_prob: float):
    
    labels = input_ids.clone()
    # mask candidates: not PAD and not special
    special = (input_ids == PAD_ID) | (input_ids == CLS_ID) | (input_ids == SEP_ID)
    candidates = (~special) & (attention_mask.bool())

    # choose mask positions
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_positions = (rand < mlm_prob) & candidates

    labels[~mask_positions] = -100  # only compute loss on masked positions

    # 80% replace with MASK
    rand2 = torch.rand(input_ids.shape, device=input_ids.device)
    mask_mask = mask_positions & (rand2 < 0.8)
    input_ids[mask_mask] = MASK_ID

    # 10% replace with random token
    rand3 = torch.rand(input_ids.shape, device=input_ids.device)
    random_mask = mask_positions & (rand2 >= 0.8) & (rand3 < 0.5)  # roughly 10%
    # random tokens in vocab range excluding specials (quick + safe)
    vocab_min = EVENT_OFFSET
    vocab_max = VOCAB_SIZE - 1
    input_ids[random_mask] = torch.randint(vocab_min, vocab_max + 1, size=input_ids[random_mask].shape, device=input_ids.device)

    # 10% keep same -> do nothing
    return input_ids, labels, mask_positions



# Datasets

class LogTokenDataset(Dataset):
    def __init__(self, X_tok: np.ndarray, y: np.ndarray | None = None):
        self.X_tok = X_tok
        self.y = y

    def __len__(self):
        return self.X_tok.shape[0]

    def __getitem__(self, idx):
        input_ids, attn = add_special_tokens_and_pad(self.X_tok[idx])
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
        if self.y is not None:
            item["labels"] = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return item



# Metrics helpers

@torch.no_grad()
def masked_token_accuracy(model_mlm, loader):
    model_mlm.eval()
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)

        # create masked version
        masked_input, labels, mask_pos = make_mlm_inputs(input_ids.clone(), attn, MLM_PROB)

        out = model_mlm(input_ids=masked_input, attention_mask=attn)
        logits = out.logits  # (B,T,V)

        pred = logits.argmax(dim=-1)
        # evaluate only masked positions
        mask = mask_pos
        if mask.any():
            correct += (pred[mask] == labels[mask]).sum().item()
            total += mask.sum().item()

    return (correct / total) if total > 0 else 0.0


@torch.no_grad()
def next_event_correlation_accuracy(model_mlm, X_tok: np.ndarray, sample_limit: int = 2000):
    
    model_mlm.eval()
    n = min(len(X_tok), sample_limit)
    correct = 0
    total = 0

    for i in range(n):
        input_ids_np, attn_np = add_special_tokens_and_pad(X_tok[i])
        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
        attn = torch.tensor(attn_np, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # last event position = 1 + (SEQ_LEN-1) unless that event is PAD
        last_pos = 1 + (SEQ_LEN - 1)
        true_token = input_ids[0, last_pos].item()
        if true_token == PAD_ID:
            continue  # skip fully padded tails

        masked = input_ids.clone()
        masked[0, last_pos] = MASK_ID

        out = model_mlm(input_ids=masked, attention_mask=attn)
        pred = out.logits[0, last_pos].argmax().item()

        correct += int(pred == true_token)
        total += 1

    return (correct / total) if total > 0 else 0.0


def save_confusion_matrix(cm, path: Path, title: str):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_roc_curves(y_true, y_prob, num_classes, path: Path, title: str):
    plt.figure(figsize=(9, 6))
    for c in range(num_classes):
        yt = (y_true == c).astype(int)
        scores = y_prob[:, c]
        thresholds = np.unique(scores)[::-1]
        tpr, fpr = [], []
        P = yt.sum()
        N = (yt == 0).sum()

        step = max(1, len(thresholds) // 500)
        for thr in thresholds[::step]:
            pred_pos = scores >= thr
            TP = (pred_pos & (yt == 1)).sum()
            FP = (pred_pos & (yt == 0)).sum()
            tpr.append(TP / P if P > 0 else 0.0)
            fpr.append(FP / N if N > 0 else 0.0)

        plt.plot(fpr, tpr, label=f"class {c}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


@torch.no_grad()
def save_attention_heatmap(model_cls, X_tok: np.ndarray, out_path: Path, sample_index: int = 0, layer: int = -1, head: int = 0):
    
    model_cls.eval()
    input_ids_np, attn_np = add_special_tokens_and_pad(X_tok[sample_index])
    input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=DEVICE).unsqueeze(0)
    attn = torch.tensor(attn_np, dtype=torch.long, device=DEVICE).unsqueeze(0)

    out = model_cls(input_ids=input_ids, attention_mask=attn, output_attentions=True, return_dict=True)
    attentions = out.attentions  # tuple: (layers) each (B, heads, T, T)

    A = attentions[layer][0, head].detach().cpu().numpy()  # (T,T)

    plt.figure(figsize=(8, 7))
    plt.imshow(A, interpolation="nearest")
    plt.title(f"Attention Heatmap (layer {layer}, head {head})")
    plt.colorbar()
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



# Main

if __name__ == "__main__":
    set_seed(SEED)

    # Load token sequences (you must have created these using KMeans step)
    X_train_tok = np.load(DATA_DIR / "X_train_tok.npy")  # (N,L) ints in [0..K-1]
    X_test_tok  = np.load(DATA_DIR / "X_test_tok.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")

    # Infer vocab size K from token data
    K = int(max(X_train_tok.max(), X_test_tok.max())) + 1
    VOCAB_SIZE = EVENT_OFFSET + K  # PAD/CLS/SEP/MASK + events

    num_classes = int(max(y_train.max(), y_test.max())) + 1

    print("DEVICE:", DEVICE)
    print("K (event vocab):", K)
    print("VOCAB_SIZE:", VOCAB_SIZE)
    print("num_classes:", num_classes)

    # Build BERT config
    bert_cfg = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        intermediate_size=HIDDEN * 4,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
        max_position_embeddings=MAX_LEN + 8,
        type_vocab_size=1,
        pad_token_id=PAD_ID,
    )

   
    # PRETRAIN (Masked Event Modeling)
   
    mlm_model = BertForMaskedLM(bert_cfg).to(DEVICE)
    opt = torch.optim.AdamW(mlm_model.parameters(), lr=LR_PRETRAIN)

    pretrain_ds = LogTokenDataset(X_train_tok, y=None)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("\n=== Pretraining MLM ===")
    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        mlm_model.train()
        total_loss = 0.0
        n_seen = 0

        for batch in pretrain_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)

            masked_input, labels, _ = make_mlm_inputs(input_ids, attn, MLM_PROB)

            opt.zero_grad()
            out = mlm_model(input_ids=masked_input, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        print(f"MLM Epoch {epoch}/{EPOCHS_PRETRAIN} loss={total_loss / max(1,n_seen):.4f}")

    # Save pretrained MLM weights
    mlm_path = MODEL_DIR / "model2_mlm_pretrained"
    mlm_model.save_pretrained(mlm_path)
    print("Saved MLM model to:", mlm_path.resolve())

    # Sequence reconstruction accuracy (masked token accuracy)
    # Use a small slice of test as "reconstruction eval"
    recon_ds = LogTokenDataset(X_test_tok[:5000], y=None)
    recon_loader = DataLoader(recon_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    recon_acc = masked_token_accuracy(mlm_model, recon_loader)

    # Correlation accuracy (next-event dependency)
    corr_acc = next_event_correlation_accuracy(mlm_model, X_test_tok, sample_limit=2000)

    print(f"Sequence reconstruction (masked-token) accuracy: {recon_acc:.4f}")
    print(f"Correlation (next-event) accuracy: {corr_acc:.4f}")

  
    # FINETUNE (Attack-stage classification)
  
    bert_cfg.num_labels = num_classes



    # Load encoder weights from MLM into classifier (transfer learning)
    cls_model = BertForSequenceClassification(bert_cfg).to(DEVICE)

    # Smoothed class weights for imbalance
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    class_counts = torch.bincount(y_train_t, minlength=num_classes).float()
    class_weights = torch.sqrt(class_counts.sum() / (class_counts + 1e-9))
    class_weights = class_weights / class_weights.sum() * num_classes
    print("Class counts:", class_counts.tolist())
    print("Smoothed class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    opt2 = torch.optim.AdamW(cls_model.parameters(), lr=LR_FINETUNE)

    train_ds = LogTokenDataset(X_train_tok, y_train)
    test_ds  = LogTokenDataset(X_test_tok, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\n=== Fine-tuning classifier ===")
    for epoch in range(1, EPOCHS_FINETUNE + 1):
        cls_model.train()
        total_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            opt2.zero_grad()
            out = cls_model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            logits = out.logits
            loss = criterion(logits, labels)
            loss.backward()
            opt2.step()

            bs = input_ids.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        print(f"CLS Epoch {epoch}/{EPOCHS_FINETUNE} loss={total_loss / max(1,n_seen):.4f}")

    # Save classifier
    cls_path = MODEL_DIR / "model2_classifier_finetuned"
    cls_model.save_pretrained(cls_path)
    print("Saved classifier to:", cls_path.resolve())


    #  EVALUATE classification + visuals

    cls_model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()

            out = cls_model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            prob = torch.softmax(out.logits, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)

            all_true.append(labels)
            all_pred.append(pred)
            all_prob.append(prob)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        roc_auc = None

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "roc_auc_macro_ovr": None if roc_auc is None else float(roc_auc),
        "sequence_reconstruction_accuracy": float(recon_acc),
        "correlation_accuracy_next_event": float(corr_acc),
    }

    print("\nFinal Metrics:", metrics)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png", "Confusion Matrix (Model 2 - BERT)")

    # ROC curves
    save_roc_curves(y_true, y_prob, num_classes, OUT_DIR / "roc_curve.png", "ROC Curves (OvR) - Model 2 (BERT)")

    # Attention heatmap (one sample)
    save_attention_heatmap(
        cls_model,
        X_test_tok,
        OUT_DIR / "attention_heatmap.png",
        sample_index=0,
        layer=-1,
        head=0
    )

    print("Saved results to:", OUT_DIR.resolve())
