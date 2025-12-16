import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


# Config

DATA_DIR = Path("data/processed")   # folder
OUT_DIR  = Path("results/model1_transformer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAD_VAL = -1.0            
SEQ_LEN = 50              
BATCH_SIZE = 256
EPOCHS = 8
LR = 1e-4
D_MODEL = 128
NHEAD = 4
NLAYERS = 2
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load data

X_train = np.load(DATA_DIR / "X_train.npy")  # (N, L, D)
y_train = np.load(DATA_DIR / "y_train.npy")  # (N,)
X_test  = np.load(DATA_DIR / "X_test.npy")
y_test  = np.load(DATA_DIR / "y_test.npy")

# Sanity
assert X_train.ndim == 3 and X_train.shape[1] == SEQ_LEN, f"Expected (N,{SEQ_LEN},D), got {X_train.shape}"
input_dim = X_train.shape[2]
num_classes = int(max(y_train.max(), y_test.max())) + 1


# Dataset that also returns pad mask
# pad_mask: True where padded (Transformer expects True=PAD)

class SeqDataset(Dataset):
    def __init__(self, X, y, pad_val=-1.0):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.pad_val = pad_val

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (L, D)
        # padded rows were created with PAD_VAL everywhere; validating first feature is enough
        pad_mask = (x[:, 0] == self.pad_val) 
        return x, self.y[idx], pad_mask

train_ds = SeqDataset(X_train, y_train, pad_val=PAD_VAL)
test_ds  = SeqDataset(X_test,  y_test,  pad_val=PAD_VAL)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)




# Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1), :]

# Transformer Encoder Classifier with PAD masking + masked mean pooling

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=4, nlayers=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos  = PositionalEncoding(d_model, max_len=512)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )

        # Disable NestedTensor to remove warning
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=nlayers,
            enable_nested_tensor=False
        )

        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def masked_mean_pool(self, h, pad_mask):
        # h: (B, L, D), pad_mask: (B, L) True where pad
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
        summed = (h * valid).sum(dim=1)            # (B, D)
        denom = valid.sum(dim=1).clamp(min=1.0)    # (B, 1)
        return summed / denom

    def forward(self, x, pad_mask):
        # x: (B, L, input_dim), pad_mask: (B, L) bool
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        pooled = self.masked_mean_pool(h, pad_mask)
        return self.cls(pooled)


# Model, Optimizer, Loss

model = TransformerClassifier(input_dim, num_classes, D_MODEL, NHEAD, NLAYERS, DROPOUT).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# Smoothed class-weighted loss (prevents over-correction)
y_train_t = torch.tensor(y_train, dtype=torch.long)
class_counts = torch.bincount(y_train_t, minlength=num_classes).float()

class_weights = torch.sqrt(class_counts.sum() / (class_counts + 1e-9))
class_weights = class_weights / class_weights.sum() * num_classes  # normalize

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
print("Class counts:", class_counts.tolist())
print("Smoothed class weights:", class_weights.tolist())


# Train

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for xb, yb, mb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        mb = mb.to(DEVICE)

        opt.zero_grad()
        logits = model(xb, mb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} loss={avg_loss:.4f}")

# Save model 

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "transformer_model.pt")
print("Saved model to:", (MODEL_DIR / "transformer_model.pt").resolve())


# Evaluate



model.eval()



all_true, all_pred, all_prob = [], [], []

with torch.no_grad():
    for xb, yb, mb in test_loader:
        xb = xb.to(DEVICE)
        mb = mb.to(DEVICE)

        logits = model(xb, mb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)

        all_prob.append(prob)
        all_pred.append(pred)
        all_true.append(yb.numpy())

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)
y_prob = np.concatenate(all_prob)

acc = accuracy_score(y_true, y_pred)

prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

# ROC-AUC macro OVR
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
}

print("Metrics:", metrics)



with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Model 1)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()

# ROC curves (one-vs-rest) 
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
plt.title("ROC Curves (One-vs-Rest) - Model 1")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png", dpi=300)
plt.close()

print("Saved to:", OUT_DIR.resolve())
