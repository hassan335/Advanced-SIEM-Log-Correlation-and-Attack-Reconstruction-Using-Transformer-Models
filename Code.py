import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import ast
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import  MinMaxScaler
pd.set_option("future.no_silent_downcasting", True)
# ---- Output directory for plots ----
PLOT_DIR = Path("results/eda")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

#Load the raw CSV and clean timestamps
# df = pd.read_csv("data/raw/advanced_siem_csv/train.csv")
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp"])

# # Build host_id (dataset doesn't have host_id directly)
# def safe_to_dict(x):
#     if isinstance(x, dict): return x
#     if isinstance(x, str):
#         try: return ast.literal_eval(x)
#         except Exception: return {}
#     return {}
#Parse advanced_metadata safely into a Python dict
# df["advanced_metadata_dict"] = df["advanced_metadata"].apply(safe_to_dict)

# Build a host_id column because dataset doesn’t have a clean host identifier
# df["host_id"] = df["device_id"]
# df["host_id"] = df["host_id"].fillna(df["advanced_metadata_dict"].apply(lambda d: d.get("device_hash")))
# df["host_id"] = df["host_id"].fillna(df["mac_address"])
# df["host_id"] = df["host_id"].fillna("unknown_host")

# # Create “log type” views
# ids_df = df[df["alert_type"].notna() | df["signature_id"].notna() | df["category"].notna() | (df["event_type"] == "ids_alert")].copy()
# firewall_df = df[df["protocol"].notna() | df["src_port"].notna() | df["dst_port"].notna() | df["bytes"].notna()].copy()
# system_df = df.drop(ids_df.index.union(firewall_df.index)).copy()

# # Save them as separate "log files"
# out_logs = Path("data/raw/split_logs")
# out_logs.mkdir(parents=True, exist_ok=True)

# firewall_df.to_csv(out_logs / "firewall.csv", index=False)
# ids_df.to_csv(out_logs / "ids.csv", index=False)
# system_df.to_csv(out_logs / "system.csv", index=False)

# print("Saved split logs to:", out_logs.resolve())
# print("Firewall:", firewall_df.shape, "IDS:", ids_df.shape, "System:", system_df.shape)



LOG_DIR = Path("data/raw/split_logs")

firewall = pd.read_csv(LOG_DIR / "firewall.csv")
ids      = pd.read_csv(LOG_DIR / "ids.csv")
system   = pd.read_csv(LOG_DIR / "system.csv")

# parse timestamp again after CSV load
for d in (firewall, ids, system):
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d.dropna(subset=["timestamp"], inplace=True)

print("Loaded:", firewall.shape, ids.shape, system.shape)


# MERGE LOGS (timestamp rounding + src/dst IP + host_id)

ROUND_TO = "1s"

def prep_keys(d):
    d = d.copy()
    d["ts_key"] = d["timestamp"].dt.round(ROUND_TO)
    d["src_ip_key"] = d.get("src_ip").fillna("NA")
    d["dst_ip_key"] = d.get("dst_ip").fillna("NA")
    d["host_id"] = d.get("host_id").fillna("unknown_host")
    return d

firewall_k = prep_keys(firewall)
ids_k      = prep_keys(ids)
system_k   = prep_keys(system)

keys = ["ts_key", "src_ip_key", "dst_ip_key", "host_id"]

unified_df = (
    firewall_k.merge(ids_k, on=keys, how="outer", suffixes=("_fw", "_ids"))
              .merge(system_k, on=keys, how="outer", suffixes=("", "_sys"))
)

unified_df["timestamp_unified"] = unified_df["ts_key"]

print("Unified shape:", unified_df.shape)

#CLEANING SETUP

dfc = unified_df.copy()



# Drop IDs / hashes (usually non-informative for modeling)
id_like_cols = [
    "event_id_fw", "event_id_ids", "event_id", "event_id_sys",
    "signature_id_fw", "signature_id_ids", "signature_id", "signature_id_sys",
    "resource_id_fw", "resource_id_ids", "resource_id", "resource_id_sys",
    "model_id_fw", "model_id_ids", "model_id", "model_id_sys",
    "input_hash_fw", "input_hash_ids", "input_hash", "input_hash_sys",
    "output_hash_fw", "output_hash_ids", "output_hash", "output_hash_sys",
]

to_drop = [c for c in id_like_cols if c in dfc.columns]
dfc.drop(columns=to_drop, inplace=True, errors="ignore")


# TIMESTAMP NORMALIZATION (UTC + UNIX)

ts_col = "timestamp_unified" if "timestamp_unified" in dfc.columns else "timestamp"
dfc[ts_col] = pd.to_datetime(dfc[ts_col], errors="coerce")

# Normalize to UTC (if naive, assume UTC)
if dfc[ts_col].dt.tz is None:
    dfc["timestamp_utc"] = dfc[ts_col].dt.tz_localize("UTC")
else:
    dfc["timestamp_utc"] = dfc[ts_col].dt.tz_convert("UTC")

dfc.dropna(subset=["timestamp_utc"], inplace=True)
dfc["timestamp_unix"] = (dfc["timestamp_utc"].astype("int64") // 10**9).astype("int64")


# DROP ROWS MISSING CORE METADATA (FIXED)

dfc["event_type_unified"] = dfc[["event_type_fw", "event_type_ids", "event_type"]].bfill(axis=1).iloc[:, 0]
dfc.dropna(subset=["timestamp_utc", "event_type_unified"], inplace=True)

#IMPUTE NaNs (median for numeric, mode for objects)

num_cols = dfc.select_dtypes(include=[np.number]).columns
dfc[num_cols] = dfc[num_cols].fillna(dfc[num_cols].median(numeric_only=True))

obj_cols = dfc.select_dtypes(include=["object"]).columns
for c in obj_cols:
    if dfc[c].isna().any():
        mode_vals = dfc[c].mode(dropna=True)
        if len(mode_vals) > 0:
            dfc[c] = dfc[c].fillna(mode_vals.iloc[0])


# INF HANDLING 

dfc = dfc.replace([np.inf, -np.inf], np.nan)
dfc = dfc.infer_objects(copy=False)
dfc.fillna(dfc.median(numeric_only=True), inplace=True)


# LABEL ENCODING starts

def coalesce(df, cols, out):
    cols = [c for c in cols if c in df.columns]
    df[out] = df[cols].bfill(axis=1).iloc[:, 0] if cols else np.nan

coalesce(dfc, ["event_type_fw", "event_type_ids", "event_type"], "event_type_u")
coalesce(dfc, ["category_fw", "category_ids", "category"], "category_u")
coalesce(dfc, ["alert_type_fw", "alert_type_ids", "alert_type"], "alert_type_u")
coalesce(dfc, ["protocol_fw", "protocol_ids", "protocol"], "protocol_u")
coalesce(dfc, ["parent_process_fw", "parent_process_ids", "parent_process"], "parent_process_u")
coalesce(dfc, ["bytes_fw", "bytes_ids", "bytes"], "bytes_u")

# keep these for labeling 
coalesce(dfc, ["description_fw", "description_ids", "description", "description_sys"], "description_u")
coalesce(dfc, ["additional_info_fw", "additional_info_ids", "additional_info", "additional_info_sys"], "additional_info_u")

# MITRE extraction
mitre_pattern = r"(T\d{4}(?:\.\d{3})?)"
dfc["mitre_technique"] = dfc["additional_info_u"].astype(str).str.extract(mitre_pattern, expand=False)
m = dfc["mitre_technique"].isna()
dfc.loc[m, "mitre_technique"] = dfc.loc[m, "description_u"].astype(str).str.extract(mitre_pattern, expand=False)

def derive_attack_stage(row):
    text = " ".join([
        str(row.get("event_type_u", "")),
        str(row.get("category_u", "")),
        str(row.get("alert_type_u", "")),
        str(row.get("protocol_u", "")),
        str(row.get("parent_process_u", "")),
        str(row.get("additional_info_u", "")),
        str(row.get("description_u", "")),
        str(row.get("mitre_technique", "")),
    ]).lower()

    text = re.sub(r"[\s\-]+", " ", text)

    b = row.get("bytes_u")
    try:
        b = float(b) if pd.notna(b) else np.nan
    except Exception:
        b = np.nan

    # Exfiltration
   # In derive_attack_stage, replace exfil condition with:
    if (pd.notna(b) and b > 2_000_000) or re.search(r"\bexfiltration\b|\bexfil\b|data leak|data theft|upload|download|transfer", text):
        return "exfiltration"

    # Lateral movement
    if re.search(r"lateral movement|rdp|smb|winrm|psexec|ssh|remote login|remote exec|pass the hash", text):
        return "lateral_movement"

    # Privilege escalation
    if re.search(r"privilege escalation|uac|sudo\b|runas|token manipulation|t1547|t1068|t1055|powershell|cmd\.exe", text):
        return "privilege_escalation"

    # Exploit
    if re.search(r"\bexploit\b|credential stuffing|sql injection|rce|cve|payload|malware|buffer overflow|brute force|zero day", text):
        return "exploit"

    # Recon
    if re.search(r"\brecon\b|reconnaissance|scan|port scan|enumeration|discovery|probe", text):
        return "recon"

    return "benign"

dfc["attack_stage"] = dfc.apply(derive_attack_stage, axis=1)

label_map = {
    "benign": 0,
    "recon": 1,
    "exploit": 2,
    "privilege_escalation": 3,
    "lateral_movement": 4,
    "exfiltration": 5
}
dfc["label"] = dfc["attack_stage"].map(label_map).astype("Int64")



drop_after_label = [
    "raw_log_fw", "raw_log_ids", "raw_log", "raw_log_sys",
    "description_fw", "description_ids", "description", "description_sys",
    "additional_info_fw", "additional_info_ids", "additional_info", "additional_info_sys",
]
dfc.drop(columns=[c for c in drop_after_label if c in dfc.columns], inplace=True, errors="ignore")




CAT_COLS = ["event_type_u", "protocol_u", "alert_type_u", "category_u", "parent_process_u"]
CAT_COLS = [c for c in CAT_COLS if c in dfc.columns]

X_cat_raw = dfc[CAT_COLS].fillna("UNK").astype(str)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_ohe = ohe.fit_transform(X_cat_raw)



NUM_COLS_CANDIDATES = [
    "timestamp_unix",
    "bytes_u",
    "duration_fw", "duration_ids", "duration", "duration_sys",
    "src_port_fw", "src_port_ids", "src_port", "src_port_sys",
    "dst_port_fw", "dst_port_ids", "dst_port", "dst_port_sys",
]

NUM_COLS = [c for c in NUM_COLS_CANDIDATES if c in dfc.columns]

# Convert to numeric
for c in NUM_COLS:
    dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

# Keep only columns that are NOT all-NaN
NUM_COLS = [c for c in NUM_COLS if dfc[c].notna().any()]

# Fill remaining NaNs 
dfc[NUM_COLS] = dfc[NUM_COLS].fillna(dfc[NUM_COLS].median(numeric_only=True))

print("Scaling these numeric columns (non-empty):", NUM_COLS)

# Choose ONE:
scaler = MinMaxScaler()       # OR StandardScaler()
scaled_cols = [c + "_scaled" for c in NUM_COLS]
dfc[scaled_cols] = scaler.fit_transform(dfc[NUM_COLS])



TIME_COL = "timestamp_utc"
HOST_COL = "host_id"

dfc[TIME_COL] = pd.to_datetime(dfc[TIME_COL], errors="coerce")

# Sort
dfc = dfc.sort_values([HOST_COL, TIME_COL]).reset_index(drop=True)

# Sessionization by gap , 30 minute
GAP_SECONDS = 30 * 60

gap = dfc.groupby(HOST_COL)[TIME_COL].diff().dt.total_seconds()

# start a new session if gap is NaN (first row) or gap > threshold
new_session = gap.isna() | (gap > GAP_SECONDS)

# session index per host
dfc["session_idx"] = new_session.groupby(dfc[HOST_COL]).cumsum().astype(int)

# session key
dfc["session_key"] = dfc[HOST_COL].astype(str) + "_S" + dfc["session_idx"].astype(str)

# Now delta-time within session 
dfc["delta_t_sec"] = dfc.groupby("session_key")[TIME_COL].diff().dt.total_seconds().fillna(0)
dfc.loc[dfc["delta_t_sec"] < 0, "delta_t_sec"] = 0

# Optional: cap extreme gaps to reduce outliers 
dfc["delta_t_sec_capped"] = dfc["delta_t_sec"].clip(upper=24 * 3600)

# Optional: log transform 
dfc["delta_t_log1p"] = np.log1p(dfc["delta_t_sec_capped"])

print(dfc[["host_id", "session_key", TIME_COL, "delta_t_sec", "delta_t_log1p"]].head(10))
print(dfc["delta_t_sec"].describe())


SEQ_LEN = 50  # 
PAD_VAL = 0.0

# Choosing event feature columns to feed to Transformer

FEATURE_COLS = []


FEATURE_COLS += [c for c in dfc.columns if c.endswith("_scaled")]
FEATURE_COLS += [c for c in ["delta_t_log1p"] if c in dfc.columns]


FEATURE_COLS += [c for c in ["event_type_u_idx", "protocol_u_idx", "category_u_idx", "alert_type_u_idx"] if c in dfc.columns]

FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))  # unique
print("Using features:", FEATURE_COLS)

# Sort events inside each session
dfc = dfc.sort_values(["session_key", "timestamp_utc"]).reset_index(drop=True)

X, y, groups = [], [], []

for sk, g in dfc.groupby("session_key"):
    feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = g["label"].to_numpy()

    # If session is longer than SEQ_LEN, chunk it into multiple sequences
    for start in range(0, len(g), SEQ_LEN):
        window = feats[start:start + SEQ_LEN]
        target = labels[min(start + SEQ_LEN - 1, len(labels) - 1)]  # label of last event

        # pad short windows
        if window.shape[0] < SEQ_LEN:
            pad = np.full((SEQ_LEN - window.shape[0], window.shape[1]), PAD_VAL, dtype=np.float32)
            window = np.vstack([window, pad])

        X.append(window)
        y.append(int(target))
        groups.append(sk)

X = np.stack(X)         
y = np.array(y)         
groups = np.array(groups)



SEQ_LEN = 50         
PAD_VAL = 0.0        
# ort events by timestamp within each session
dfc = dfc.sort_values(["session_key", "timestamp_utc"]).reset_index(drop=True)

# 2)hoose features per event
FEATURE_COLS = [
    "timestamp_unix_scaled",
    "bytes_u_scaled",
    "duration_fw_scaled",
    "src_port_fw_scaled",
    "dst_port_fw_scaled",
    "delta_t_log1p",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in dfc.columns]

X, y, groups = [], [], []

# Slice sequences of equal length + pad short sequences
for sk, g in dfc.groupby("session_key"):
    feats = g[FEATURE_COLS].to_numpy(dtype=np.float32)
    labels = g["label"].to_numpy()

    # split into chunks of SEQ_LEN
    for start in range(0, len(g), SEQ_LEN):
        window = feats[start:start + SEQ_LEN]
        target = labels[min(start + SEQ_LEN - 1, len(labels) - 1)]  # label of last event in window

        # pad if shorter than SEQ_LEN
        if window.shape[0] < SEQ_LEN:
            pad = np.full((SEQ_LEN - window.shape[0], window.shape[1]), PAD_VAL, dtype=np.float32)
            window = np.vstack([window, pad])

        X.append(window)
        y.append(int(target))
        groups.append(sk)

X = np.stack(X)       
y = np.array(y)        
groups = np.array(groups)






gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]




#  Log Source Distribution 
source_counts = pd.Series({
    "firewall": len(firewall),
    "ids": len(ids),
    "system": len(system)
}).sort_values(ascending=False)

print(source_counts)

plt.figure(figsize=(6, 4))
source_counts.plot(kind="bar")
plt.title("Log Source Distribution")
plt.xlabel("Log Source")
plt.ylabel("Number of Events")
plt.tight_layout()

# ---- Save plot ----
out_path = PLOT_DIR / "log_source_distribution.png"
plt.savefig(out_path, dpi=300)
plt.show()



# Choose the best column for event type 
event_col = "event_type_u" if "event_type_u" in dfc.columns else "event_type"

top20 = dfc[event_col].astype(str).value_counts().head(20)

plt.figure(figsize=(10, 5))
top20.plot(kind="bar")
plt.title("Top 20 Event Types")
plt.xlabel("Event Type")
plt.ylabel("Number of Events")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_path = PLOT_DIR / "event_type_top20.png"
plt.savefig(out_path, dpi=300)
plt.show()



# Prefer attack_stage (string) if available, else fall back to numeric label
if "attack_stage" in dfc.columns:
    stage_counts = dfc["attack_stage"].astype(str).value_counts()
    title = "Attack Stage Distribution"
    xlab = "Attack Stage"
else:
    stage_counts = dfc["label"].value_counts().sort_index()
    title = "Attack Stage Distribution (Numeric Labels)"
    xlab = "Label"

# print(stage_counts)

plt.figure(figsize=(8, 4))
stage_counts.plot(kind="bar")
plt.title(title)
plt.xlabel(xlab)
plt.ylabel("Number of Events")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_path = PLOT_DIR / "attack_stage_distribution.png"
plt.savefig(out_path, dpi=300)
plt.show()





np.save(OUT/"X_train.npy", X_train)
np.save(OUT/"y_train.npy", y_train)
np.save(OUT/"X_test.npy",  X_test)
np.save(OUT/"y_test.npy",  y_test)

print("Saved arrays to:", OUT.resolve())










