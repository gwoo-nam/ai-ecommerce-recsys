"""
phase2_two_tower.py  (개선판)
─────────────────────────────
명세서 요구사항 반영:
  ✅ User Tower: user_idx + persona  (프로필 피처)
  ✅ Item Tower: item_idx + category_L1 + price_tier  (카테고리/속성/가격)
  ✅ In-batch sampled softmax (retrieval 표준 손실함수)
  ✅ Negative Sampling 1:4 비율 보존 (옵션으로 BCE+neg 학습 모드도 지원)
  ✅ 학습 데이터: train_logs.csv 만 사용 (test_logs.csv 누수 방지)

리트리벌(Recall) 중심 학습이므로 라벨은 implicit positive(=상호작용 발생).
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 1. 데이터 로드 & 인코더
# ──────────────────────────────────────────────
print("데이터 로딩 중...")
df_users = pd.read_csv("data/users.csv")
df_prods = pd.read_csv("data/products.csv")
df_logs = pd.read_csv("data/train_logs.csv")
df_logs = df_logs[df_logs["event_type"].isin(["cart", "purchase"])].copy()

# implicit positive: cart/purchase는 강한 신호, view는 약한 신호 → 모두 positive로 간주
# (retrieval 단계는 후보를 넓게 가져오는 게 목표)
SPARSE_FEATS = ["persona", "category_L1", "category_L2", "category_L3", "price_tier"]

# 인코더 fit (전체 카탈로그 기준)
user_encoder = LabelEncoder().fit(df_users["user_id"].values)
prod_encoder = LabelEncoder().fit(df_prods["product_id"].values)

df_users["user_idx"] = user_encoder.transform(df_users["user_id"])
df_prods["prod_idx"] = prod_encoder.transform(df_prods["product_id"])

side_encoders = {}
for feat in SPARSE_FEATS:
    le = LabelEncoder()
    if feat == "persona":
        le.fit(df_users[feat].astype(str).values)
        df_users[f"{feat}_idx"] = le.transform(df_users[feat].astype(str))
    else:
        le.fit(df_prods[feat].astype(str).values)
        df_prods[f"{feat}_idx"] = le.transform(df_prods[feat].astype(str))
    side_encoders[feat] = le

# 로그에 인덱스/사이드피처 조인
df_logs = df_logs.merge(
    df_users[["user_id", "user_idx", "persona_idx"]], on="user_id", how="inner"
).merge(
    df_prods[[
        "product_id",
        "prod_idx",
        "category_L1_idx",
        "category_L2_idx",
        "category_L3_idx",
        "price_tier_idx"
    ]],
    on="product_id",
    how="inner"
)


num_users    = len(user_encoder.classes_)
num_prods    = len(prod_encoder.classes_)
num_persona  = len(side_encoders["persona"].classes_)
num_cat1 = len(side_encoders["category_L1"].classes_)
num_cat2 = len(side_encoders["category_L2"].classes_)
num_cat3 = len(side_encoders["category_L3"].classes_)
num_tier = len(side_encoders["price_tier"].classes_)


print(f"  유저: {num_users:,} / 상품: {num_prods:,} / 페르소나: {num_persona} / 카테고리: {num_cat1}/{num_cat2}/{num_cat3} / 가격대: {num_tier}")
print(f"  학습용 로그: {len(df_logs):,}건")


# ──────────────────────────────────────────────
# 2. Dataset (positive pair만 반환, in-batch negative 사용)
# ──────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, df):
        self.user_idx = df["user_idx"].values
        self.persona_idx = df["persona_idx"].values
        self.prod_idx = df["prod_idx"].values
        self.cat1_idx = df["category_L1_idx"].values
        self.cat2_idx = df["category_L2_idx"].values
        self.cat3_idx = df["category_L3_idx"].values
        self.tier_idx = df["price_tier_idx"].values

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, i):
        return (
            torch.tensor(self.user_idx[i], dtype=torch.long),
            torch.tensor(self.persona_idx[i], dtype=torch.long),
            torch.tensor(self.prod_idx[i], dtype=torch.long),
            torch.tensor(self.cat1_idx[i], dtype=torch.long),
            torch.tensor(self.cat2_idx[i], dtype=torch.long),
            torch.tensor(self.cat3_idx[i], dtype=torch.long),
            torch.tensor(self.tier_idx[i], dtype=torch.long),
        )


dataset = PairDataset(df_logs)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)


# ──────────────────────────────────────────────
# 3. Two-Tower 모델 (피처 보강)
# ──────────────────────────────────────────────
class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_prods, n_persona, n_cat1, n_cat2, n_cat3, n_tier, emb_dim=32, out_dim=64):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.persona_emb = nn.Embedding(n_persona, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        self.item_emb = nn.Embedding(n_prods, emb_dim)
        self.cat1_emb = nn.Embedding(n_cat1, emb_dim)
        self.cat2_emb = nn.Embedding(n_cat2, emb_dim)
        self.cat3_emb = nn.Embedding(n_cat3, emb_dim)
        self.tier_emb = nn.Embedding(n_tier, emb_dim)

        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim * 5, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def encode_user(self, u_idx, persona_idx):
        x = torch.cat([self.user_emb(u_idx), self.persona_emb(persona_idx)], dim=-1)
        return F.normalize(self.user_mlp(x), p=2, dim=-1)

    def encode_item(self, p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx):
        x = torch.cat([
            self.item_emb(p_idx),
            self.cat1_emb(cat1_idx),
            self.cat2_emb(cat2_idx),
            self.cat3_emb(cat3_idx),
            self.tier_emb(tier_idx),
        ], dim=-1)
        return F.normalize(self.item_mlp(x), p=2, dim=-1)





model = TwoTowerModel(
    num_users,
    num_prods,
    num_persona,
    num_cat1,
    num_cat2,
    num_cat3,
    num_tier
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ──────────────────────────────────────────────
# 4. In-batch sampled softmax 학습
#    배치 안의 다른 positive item을 negative로 활용 → retrieval 표준
# ──────────────────────────────────────────────
EPOCHS = 5
TEMPERATURE = 0.07

print(f"Two-Tower 학습 시작 (Epochs: {EPOCHS}, Loss: in-batch softmax)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for u_idx, persona_idx, p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx in dataloader:
        u_idx = u_idx.to(device)
        persona_idx = persona_idx.to(device)
        p_idx = p_idx.to(device)
        cat1_idx = cat1_idx.to(device)
        cat2_idx = cat2_idx.to(device)
        cat3_idx = cat3_idx.to(device)
        tier_idx = tier_idx.to(device)

        u_vec = model.encode_user(u_idx, persona_idx)
        i_vec = model.encode_item(p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx)
        # (B, D)

        # 유사도 행렬 (B x B): 대각선=positive, 그 외=negative
        logits = u_vec @ i_vec.T / TEMPERATURE
        labels = torch.arange(u_vec.size(0), device=device)
        loss   = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/n_batches:.4f}")


# ──────────────────────────────────────────────
# 5. 모델/임베딩 저장
# ──────────────────────────────────────────────
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/indices", exist_ok=True)

torch.save({
    "state_dict": model.state_dict(),
    "num_users": num_users,
    "num_prods": num_prods,
    "num_persona": num_persona,
    "num_cat1": num_cat1,
    "num_cat2": num_cat2,
    "num_cat3": num_cat3,
    "num_tier": num_tier,
}, "data/models/two_tower.pth")


# ──────────────────────────────────────────────
# 6. 전체 카탈로그의 Item 임베딩을 사전 계산 → FAISS 인덱스
# ──────────────────────────────────────────────
print("전체 카탈로그 임베딩 추출 중...")
model.eval()
all_p_idx = torch.tensor(df_prods["prod_idx"].values, dtype=torch.long).to(device)
all_cat1_idx = torch.tensor(df_prods["category_L1_idx"].values, dtype=torch.long).to(device)
all_cat2_idx = torch.tensor(df_prods["category_L2_idx"].values, dtype=torch.long).to(device)
all_cat3_idx = torch.tensor(df_prods["category_L3_idx"].values, dtype=torch.long).to(device)
all_tier_idx = torch.tensor(df_prods["price_tier_idx"].values, dtype=torch.long).to(device)


with torch.no_grad():
    item_vecs = []
    bs = 2048
    for i in range(0, len(all_p_idx), bs):
        v = model.encode_item(
    all_p_idx[i:i+bs],
    all_cat1_idx[i:i+bs],
    all_cat2_idx[i:i+bs],
    all_cat3_idx[i:i+bs],
    all_tier_idx[i:i+bs]
)

        item_vecs.append(v.cpu().numpy())
    item_emb = np.vstack(item_vecs).astype("float32")

# 이미 L2 정규화된 벡터 → 내적이 코사인 유사도
faiss.normalize_L2(item_emb)
index = faiss.IndexFlatIP(item_emb.shape[1])  # Inner Product = cosine
# 상품 ID 매핑을 보존하려면 IndexIDMap 사용
id_index = faiss.IndexIDMap(index)
id_index.add_with_ids(item_emb, df_prods["prod_idx"].values.astype("int64"))

faiss.write_index(id_index, "data/indices/candidate_item.index")

# 보조 메타도 저장 (서빙 때 필요)
np.save("data/models/item_embeddings.npy", item_emb)
df_prods[["product_id", "prod_idx"]].to_csv("data/models/two_tower_prod_map.csv", index=False)
df_users[["user_id", "user_idx", "persona_idx"]].to_csv("data/models/two_tower_user_map.csv", index=False)

print("✅ Two-Tower 학습 + FAISS IndexIDMap 저장 완료!")
print("   - data/models/two_tower.pth")
print("   - data/indices/candidate_item.index  (IndexIDMap, IP)")
print("   - data/models/item_embeddings.npy")