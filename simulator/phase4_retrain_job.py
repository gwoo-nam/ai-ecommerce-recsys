"""
phase4_retrain_job.py
──────────────────────
CT 파이프라인(ct_pipeline.py)이 신규 로그 1만 건을 감지하면 이 파일을 서브프로세스로 실행합니다.

수행 순서
  1. 기존 train_logs.csv 로 인코더/차원 복원
  2. new_train_logs.csv 로 DeepFM 파인튜닝
  3. 파인튜닝 완료 후 실측 지표 계산
       - DeepFM AUC / HitRate@50 / NDCG@50 / Coverage
       - CLIP+FAISS MRR·NDCG@10 / Two-Tower Recall@300 (search_metrics.json 에서 읽기)
  4. 모든 지표를 data/metrics.json 으로 저장 → 대시보드가 바로 읽음
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [Retrain Job] CT 파이프라인 워커 노드 가동 (Device: {device})")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
DATA_DIR           = "data"
USERS_CSV          = os.path.join(DATA_DIR, "users.csv")
PRODS_CSV          = os.path.join(DATA_DIR, "products.csv")
OLD_LOGS_CSV       = os.path.join(DATA_DIR, "train_logs.csv")
NEW_LOGS_CSV       = os.path.join(DATA_DIR, "new_train_logs.csv")
MODEL_PATH         = os.path.join(DATA_DIR, "models", "deepfm.pth")
METRICS_JSON       = os.path.join(DATA_DIR, "metrics.json")
SEARCH_METRICS_JSON = os.path.join(DATA_DIR, "search_metrics.json")

# ──────────────────────────────────────────────
# 1. DeepFM 모델 아키텍처 (API 서버와 완전 동일)
# ──────────────────────────────────────────────
class DeepFM(nn.Module):
    def __init__(self, feature_dims: dict, embedding_dim: int = 16):
        super().__init__()
        self.sparse_features = list(feature_dims.keys())
        self.embeddings = nn.ModuleDict(
            {feat: nn.Embedding(dim, embedding_dim) for feat, dim in feature_dims.items()}
        )
        self.linear = nn.ModuleDict(
            {feat: nn.Embedding(dim, 1) for feat, dim in feature_dims.items()}
        )
        self.bias = nn.Parameter(torch.zeros(1))
        mlp_in = len(feature_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = [self.embeddings[f](x[:, i]) for i, f in enumerate(self.sparse_features)]
        lin = self.bias + sum(self.linear[f](x[:, i]) for i, f in enumerate(self.sparse_features))
        stk = torch.stack(emb, dim=1)
        fm  = 0.5 * torch.sum(stk.sum(1) ** 2 - (stk ** 2).sum(1), dim=1, keepdim=True)
        dnn = self.mlp(torch.cat(emb, dim=1))
        return torch.sigmoid(lin + fm + dnn).squeeze()


class LogDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# ──────────────────────────────────────────────
# 2. 데이터 로드 및 인코더 복원
# ──────────────────────────────────────────────
print("1️⃣  데이터 및 기존 인코더 복원 중...")
df_users    = pd.read_csv(USERS_CSV)
df_prods    = pd.read_csv(PRODS_CSV)
df_logs_old = pd.read_csv(OLD_LOGS_CSV)
df_new_logs = pd.read_csv(NEW_LOGS_CSV)

SPARSE_FEATURES = ["user_id", "product_id", "persona", "category_L1", "price_tier"]

encoders: dict = {}
temp_merged = (
    df_logs_old
    .merge(df_users, on="user_id")
    .merge(df_prods, on="product_id")
)
for feat in SPARSE_FEATURES:
    le = LabelEncoder()
    temp_merged[feat] = le.fit_transform(temp_merged[feat].astype(str))
    encoders[feat] = le

feature_dims = {feat: len(encoders[feat].classes_) for feat in SPARSE_FEATURES}


def encode_df(df_merged: pd.DataFrame) -> np.ndarray:
    """DataFrame 을 인코더로 정수 행렬로 변환 (미등록 값은 0 처리)."""
    X = np.zeros((len(df_merged), len(SPARSE_FEATURES)), dtype=int)
    for i, feat in enumerate(SPARSE_FEATURES):
        classes = encoders[feat].classes_
        vals    = df_merged[feat].astype(str).values
        mask    = np.isin(vals, classes)
        X[mask, i] = encoders[feat].transform(vals[mask])
    return X


# ──────────────────────────────────────────────
# 3. 신규 로그 전처리 및 DataLoader
# ──────────────────────────────────────────────
print("2️⃣  신규 트렌드 데이터 전처리 중...")
df_merged_new = (
    df_new_logs
    .merge(df_users, on="user_id")
    .merge(df_prods, on="product_id")
)
df_merged_new["label"] = (
    df_merged_new["event_type"].isin(["cart", "purchase"]).astype(float)
)

X_new = encode_df(df_merged_new)
y_new = df_merged_new["label"].values

# train / eval 분리 (9:1)
n_total  = len(y_new)
n_train  = int(n_total * 0.9)
X_train, y_train = X_new[:n_train], y_new[:n_train]
X_eval,  y_eval  = X_new[n_train:], y_new[n_train:]

train_loader = DataLoader(LogDataset(X_train, y_train), batch_size=512, shuffle=True)
eval_loader  = DataLoader(LogDataset(X_eval,  y_eval),  batch_size=512, shuffle=False)


# ──────────────────────────────────────────────
# 4. 기존 모델 로드 및 파인튜닝
# ──────────────────────────────────────────────
print("3️⃣  기존 DeepFM 모델 로드 및 파인튜닝 시작...")
model = DeepFM(feature_dims).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("   기존 가중치 로드 완료.")
else:
    print("   ⚠️  기존 모델 없음 → 처음부터 학습.")

criterion = nn.BCELoss()
# 파인튜닝이므로 낮은 LR 사용
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

FINETUNE_EPOCHS = 10
model.train()
for epoch in range(FINETUNE_EPOCHS):
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"   Epoch {epoch + 1}/{FINETUNE_EPOCHS}  Loss: {total_loss / len(train_loader):.4f}")

# ──────────────────────────────────────────────
# 5. 모델 가중치 저장
# ──────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾  새 가중치 저장 완료 → {MODEL_PATH}")


# ──────────────────────────────────────────────
# 6. 실측 지표 계산
# ──────────────────────────────────────────────
print("4️⃣  오프라인 지표 실측 계산 중...")


def compute_auc(model: nn.Module, loader: DataLoader) -> float:
    """eval_loader 전체를 추론해 AUC 를 반환합니다."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds = model(X_b.to(device)).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())
    # 정답이 1종류뿐이면 AUC 계산 불가 → 0.5 반환
    unique_labels = set(all_labels)
    if len(unique_labels) < 2:
        return 0.5
    return float(roc_auc_score(all_labels, all_preds))


def compute_hitrate_ndcg(model: nn.Module, loader: DataLoader, k: int = 50) -> dict:
    """
    배치 안에서 DeepFM 예측 점수를 기준으로 상위 k 개를 선정했을 때
    실제 정답(label=1) 이 포함되는 비율(HitRate@k) 과 NDCG@k 를 반환합니다.
    """
    model.eval()
    hits, ndcg_sum, total = 0, 0.0, 0

    with torch.no_grad():
        for X_b, y_b in loader:
            preds  = model(X_b.to(device)).cpu().numpy()
            labels = y_b.numpy()

            for i in range(len(labels)):
                if labels[i] != 1.0:
                    continue
                total += 1
                # 배치 내 모든 아이템에 대해 이 아이템의 상대 순위
                rank = int(np.sum(preds >= preds[i]))  # 동점 포함 보수적 순위
                if rank <= k:
                    hits += 1
                    ndcg_sum += 1.0 / np.log2(rank + 1)

    return {
        "hitrate": hits / total if total > 0 else 0.0,
        "ndcg":    ndcg_sum / total if total > 0 else 0.0,
    }


def compute_coverage(model: nn.Module, n_users_sample: int = 200, top_k: int = 10) -> float:
    """
    샘플 유저들에게 추천된 고유 상품 수 / 전체 상품 수 로 Coverage 를 계산합니다.
    """
    model.eval()
    sample_users = df_users["user_id"].sample(
        min(n_users_sample, len(df_users)), random_state=42
    ).values

    recommended: set = set()

    with torch.no_grad():
        for uid in sample_users:
            user_info = df_users[df_users["user_id"] == uid]
            if user_info.empty:
                continue
            persona = user_info["persona"].values[0]

            # 후보 100개 랜덤 샘플
            cands = df_prods.sample(100, random_state=0).copy()
            cands["user_id"] = uid
            cands["persona"] = persona

            X = encode_df(cands)
            X_t = torch.tensor(X, dtype=torch.long).to(device)
            scores = model(X_t).cpu().numpy()

            top_idx = np.argsort(scores)[-top_k:]
            top_pids = cands.iloc[top_idx]["product_id"].tolist()
            recommended.update(top_pids)

    return len(recommended) / len(df_prods) if len(df_prods) > 0 else 0.0


def measure_recommend_latency(model: nn.Module) -> float:
    """
    단건 추천 추론 레이턴시를 p95 기준으로 측정합니다 (ms).
    """
    model.eval()
    dummy = torch.zeros(1, len(SPARSE_FEATURES), dtype=torch.long).to(device)

    latencies = []
    # warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    for _ in range(100):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.percentile(latencies, 95))


# 지표 계산 실행
auc_val     = compute_auc(model, eval_loader)
hr_result   = compute_hitrate_ndcg(model, eval_loader, k=50)
coverage    = compute_coverage(model)
rec_latency = measure_recommend_latency(model)

print(f"   DeepFM AUC       : {auc_val:.4f}  (목표 >= 0.70)")
print(f"   HitRate@50       : {hr_result['hitrate']:.4f}  (목표 >= 0.20)")
print(f"   NDCG@50          : {hr_result['ndcg']:.4f}  (목표 >= 0.08)")
print(f"   Coverage         : {coverage:.4f}  (목표 >= 0.20)")
print(f"   Rec Latency p95  : {rec_latency:.1f} ms  (목표 <= 200ms)")


# ──────────────────────────────────────────────
# 7. search_metrics.json 읽어서 최종 metrics.json 생성
# ──────────────────────────────────────────────
print("5️⃣  최종 metrics.json 생성 중...")

# search_metrics.json 이 있으면 실측값 사용, 없으면 최적 플래그로 표시
if os.path.exists(SEARCH_METRICS_JSON):
    with open(SEARCH_METRICS_JSON, "r", encoding="utf-8") as f:
        sm = json.load(f)
    mrr_val        = sm.get("mrr",        0.0)
    ndcg_10_val    = sm.get("ndcg_10",    0.0)
    recall_300_val = sm.get("recall_300", 0.0)
    search_lat     = sm.get("latency_ms", 9999)
    print(f"   ✅ search_metrics.json 로드 완료 (MRR={mrr_val:.4f})")
else:
    print(f"   ⚠️  {SEARCH_METRICS_JSON} 없음 → phase4_offline_eval.py 를 먼저 실행하세요.")
    print("      지금은 검색 지표 0 으로 기록합니다.")
    mrr_val        = 0.0
    ndcg_10_val    = 0.0
    recall_300_val = 0.0
    search_lat     = 9999

final_metrics = {
    "search": {
        "mrr":        round(mrr_val,     4),
        "ndcg_10":    round(ndcg_10_val, 4),
        "latency_ms": round(search_lat,  1),
    },
    "recommend": {
        "recall_300":  round(recall_300_val,        4),
        "auc":         round(auc_val,               4),
        "hitrate_50":  round(hr_result["hitrate"],  4),
        "ndcg_50":     round(hr_result["ndcg"],     4),
        "coverage":    round(coverage,              4),
        "latency_ms":  round(rec_latency,           1),
    },
}

os.makedirs(DATA_DIR, exist_ok=True)
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=4, ensure_ascii=False)

print(f"✅  metrics.json 저장 완료 → {METRICS_JSON}")
print("\n" + "=" * 50)
print("🏁 Retrain Job 완료!")
print(f"   AUC={final_metrics['recommend']['auc']}  "
      f"HitRate@50={final_metrics['recommend']['hitrate_50']}  "
      f"Coverage={final_metrics['recommend']['coverage']}")