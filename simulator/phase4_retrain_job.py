"""
phase4_retrain_job.py  (간소화판)
─────────────────────────────────
역할 분리:
  - 이 파일은 신규 로그로 DeepFM을 fine-tuning만 담당
  - 모든 지표 측정은 phase4_offline_eval.py가 담당
  - 두 개를 분리하면 평가 로직이 한 곳에서 관리됨 (단일 진실 원천)

CT 흐름 (ct_pipeline.py가 호출):
  1. new_train_logs.csv → DeepFM fine-tune
  2. data/models/deepfm.pth 갱신
  3. phase4_offline_eval.py 실행 → metrics.json 갱신
  4. (ct_pipeline.py가) /api/reload-model 호출
"""
import os
import sys
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [Retrain Job] 시작 (Device: {device})")

DATA_DIR     = "data"
USERS_CSV    = f"{DATA_DIR}/users.csv"
PRODS_CSV    = f"{DATA_DIR}/products.csv"
OLD_LOGS_CSV = f"{DATA_DIR}/train_logs.csv"
NEW_LOGS_CSV = f"{DATA_DIR}/new_train_logs.csv"
MODEL_PATH   = f"{DATA_DIR}/models/deepfm.pth"


# ──────────────────────────────────────────────
# DeepFM (학습 코드와 동일)
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


class LogDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.float32)


# ──────────────────────────────────────────────
# 1. 인코더 복원 (기존 train_logs로 fit)
# ──────────────────────────────────────────────
print("1️⃣  인코더 복원 중...")
df_users = pd.read_csv(USERS_CSV)
df_prods = pd.read_csv(PRODS_CSV)
df_old   = pd.read_csv(OLD_LOGS_CSV)
df_new   = pd.read_csv(NEW_LOGS_CSV)

SPARSE = ['user_id', 'product_id', 'persona', 'category_L1', 'category_L2', 'category_L3', 'price_tier']
old_full = df_old.merge(df_users, on="user_id").merge(df_prods, on="product_id")

encoders = {}
for f in SPARSE:
    le = LabelEncoder()
    le.fit(old_full[f].astype(str))
    encoders[f] = le

feat_dims = {f: len(encoders[f].classes_) for f in SPARSE}


def encode_oov(df):
    X = np.zeros((len(df), len(SPARSE)), dtype=int)
    for i, f in enumerate(SPARSE):
        cls = encoders[f].classes_
        v = df[f].astype(str).values
        m = np.isin(v, cls)
        X[m, i] = encoders[f].transform(v[m])
    return X


# ──────────────────────────────────────────────
# 2. 신규 로그 라벨링 + 인코딩
# ──────────────────────────────────────────────
print("2️⃣  신규 로그 전처리...")
new_full = df_new.merge(df_users, on="user_id").merge(df_prods, on="product_id")
new_full = new_full[new_full["event_type"].isin(["view", "cart", "purchase"])]
new_full["label"] = new_full["event_type"].isin(["cart", "purchase"]).astype(float)

X = encode_oov(new_full)
y = new_full["label"].values
print(f"   샘플: {len(y):,}건 (pos={int(y.sum()):,}, neg={int((1-y).sum()):,})")

loader = DataLoader(LogDS(X, y), batch_size=512, shuffle=True)


# ──────────────────────────────────────────────
# 3. 기존 가중치 로드 + 파인튜닝
# ──────────────────────────────────────────────
print("3️⃣  파인튜닝...")
model = DeepFM(feat_dims).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("   기존 가중치 로드")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # 낮은 LR

EPOCHS = 5
model.train()
for ep in range(EPOCHS):
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"   Epoch {ep+1}/{EPOCHS}  Loss: {total/len(loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"💾  저장: {MODEL_PATH}")


# ──────────────────────────────────────────────
# 4. 평가는 phase4_offline_eval.py에 위임 (단일 진실 원천)
# ──────────────────────────────────────────────
print("4️⃣  오프라인 평가 실행 중...")
try:
    subprocess.run([sys.executable, "phase4_offline_eval.py"], check=True)
    print("✅ Retrain + Evaluation 완료")
except subprocess.CalledProcessError as e:
    print(f"⚠️ 평가 스크립트 실행 실패: {e}")
    print("   metrics.json이 갱신되지 않았을 수 있습니다.")