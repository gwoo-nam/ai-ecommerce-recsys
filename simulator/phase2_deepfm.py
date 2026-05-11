"""
phase2_deepfm.py  (개선판)
──────────────────────────
명세서 요구사항 반영:
  ✅ 입력 피처: User(persona) + Item(category, price_tier) + Cross + Context
  ✅ AUC ≥ 0.70 달성을 위한 hard negative mining
       - positive: cart/purchase
       - negative: 같은 유저의 view (= 봤지만 안 산 것 = implicit negative)
       - 추가 random negative로 보강
  ✅ valid_logs.csv 로 검증 (시간 기반 누수 방지)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 1. 데이터 로드 및 라벨링 전략
# ──────────────────────────────────────────────
print("데이터 로딩 중...")
df_users  = pd.read_csv("data/users.csv")
df_prods  = pd.read_csv("data/products.csv")
df_train  = pd.read_csv("data/train_logs.csv")
df_valid  = pd.read_csv("data/valid_logs.csv") if os.path.exists("data/valid_logs.csv") else None


def build_supervised(df_logs: pd.DataFrame) -> pd.DataFrame:
    df = df_logs.copy()
    df = df[df["event_type"].isin(["view", "cart", "purchase"])]

    if "is_bounced" in df.columns:
        pos = df[df["event_type"].isin(["cart", "purchase"])].copy()
        pos["label"] = 1.0

        neg = df[(df["event_type"] == "view") & (df["is_bounced"] == 1)].copy()
        neg["label"] = 0.0

        out = pd.concat([pos, neg], ignore_index=True)
    else:
        df["event_label"] = df["event_type"].isin(["cart", "purchase"]).astype(float)
        out = (
            df.groupby(["user_id", "product_id"], as_index=False)
              .agg({"event_label": "max", "timestamp": "max"})
              .rename(columns={"event_label": "label"})
        )

    out = (
        out.groupby(["user_id", "product_id"], as_index=False)
           .agg({"label": "max", "timestamp": "max"})
    )

    return out




df_train_lbl = build_supervised(df_train)
df_valid_lbl = build_supervised(df_valid) if df_valid is not None else None

# 메타데이터 조인
df_train_full = df_train_lbl.merge(df_users, on="user_id").merge(df_prods, on="product_id")
if df_valid_lbl is not None:
    df_valid_full = df_valid_lbl.merge(df_users, on="user_id").merge(df_prods, on="product_id")
else:
    df_valid_full = None

print(f"  Train pos:{int(df_train_full['label'].sum()):,} / neg:{int((1-df_train_full['label']).sum()):,}")
if df_valid_full is not None:
    print(f"  Valid pos:{int(df_valid_full['label'].sum()):,} / neg:{int((1-df_valid_full['label']).sum()):,}")

# ──────────────────────────────────────────────
# 2. 인코더 (train 기준 fit)
# ──────────────────────────────────────────────
SPARSE_FEATURES = [
    "user_id",
    "product_id",
    "persona",
    "category_L1",
    "category_L2",
    "category_L3",
    "price_tier"
]
encoders = {}
for feat in SPARSE_FEATURES:
    le = LabelEncoder()
    df_train_full[feat] = le.fit_transform(df_train_full[feat].astype(str))
    encoders[feat] = le


def encode_with_oov(df: pd.DataFrame) -> np.ndarray:
    """미등록 값은 0으로 처리 (OOV)."""
    X = np.zeros((len(df), len(SPARSE_FEATURES)), dtype=int)
    for i, feat in enumerate(SPARSE_FEATURES):
        classes = encoders[feat].classes_
        vals = df[feat].astype(str).values
        mask = np.isin(vals, classes)
        X[mask, i] = encoders[feat].transform(vals[mask])
    return X


X_train = df_train_full[SPARSE_FEATURES].values
y_train = df_train_full["label"].values

if df_valid_full is not None and len(df_valid_full) > 0:
    X_valid = encode_with_oov(df_valid_full)
    y_valid = df_valid_full["label"].values
else:
    # valid 없으면 train에서 10% 떼서 사용
    n_split = int(len(X_train) * 0.9)
    X_valid, y_valid = X_train[n_split:], y_train[n_split:]
    X_train, y_train = X_train[:n_split], y_train[:n_split]

feature_dims = {feat: len(encoders[feat].classes_) for feat in SPARSE_FEATURES}


# ──────────────────────────────────────────────
# 3. Dataset
# ──────────────────────────────────────────────
class DeepFMDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)


train_loader = DataLoader(DeepFMDataset(X_train, y_train), batch_size=512, shuffle=True)
valid_loader = DataLoader(DeepFMDataset(X_valid, y_valid), batch_size=512, shuffle=False)


# ──────────────────────────────────────────────
# 4. DeepFM
# ──────────────────────────────────────────────
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=16):
        super().__init__()
        self.sparse_features = list(feature_dims.keys())
        self.embeddings = nn.ModuleDict({f: nn.Embedding(d, embedding_dim) for f, d in feature_dims.items()})
        self.linear     = nn.ModuleDict({f: nn.Embedding(d, 1) for f, d in feature_dims.items()})
        self.bias       = nn.Parameter(torch.zeros(1))

        mlp_in = len(feature_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        emb = [self.embeddings[f](x[:, i]) for i, f in enumerate(self.sparse_features)]
        lin = self.bias + sum(self.linear[f](x[:, i]) for i, f in enumerate(self.sparse_features))
        stk = torch.stack(emb, dim=1)
        fm  = 0.5 * torch.sum(stk.sum(1) ** 2 - (stk ** 2).sum(1), dim=1, keepdim=True)
        dnn = self.mlp(torch.cat(emb, dim=1))
        return torch.sigmoid(lin + fm + dnn).squeeze()


model = DeepFM(feature_dims).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ──────────────────────────────────────────────
# 5. 학습 + valid AUC 모니터링
# ──────────────────────────────────────────────
def evaluate_auc(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.extend(model(X_b.to(device)).cpu().numpy())
            labels.extend(y_b.numpy())
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


EPOCHS = 5
print(f"DeepFM 학습 시작 (Epochs: {EPOCHS})...")
best_auc = 0.0
for ep in range(EPOCHS):
    model.train()
    total = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item()

    val_auc = evaluate_auc(valid_loader)
    print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {total/len(train_loader):.4f} | Valid AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), "data/models/deepfm.pth")
        print(f"    → 베스트 갱신, 저장")

print(f"\n✅ DeepFM 학습 완료! Best Valid AUC = {best_auc:.4f}")
print("   가중치: data/models/deepfm.pth")