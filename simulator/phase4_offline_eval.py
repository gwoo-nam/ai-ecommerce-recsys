"""
phase4_offline_eval.py  (전면 개편판)
─────────────────────────────────────
명세서 요구사항 모두 반영한 정식 평가:

  ★ 데이터 누수 방지: 평가는 반드시 test_logs.csv만 사용
  ★ 진짜 추천 평가 방식:
      - Recall@300 : Two-Tower로 후보 300개 안에 정답이 있는가
      - HitRate@50, NDCG@50 : Two-Tower로 뽑은 후보를 DeepFM으로 재정렬한
                              Top-50 안에 정답이 있는가  (Multi-Stage 평가!)
      - AUC        : DeepFM이 view(neg) vs cart/purchase(pos) 를 구분하는가
      - Coverage   : 샘플 유저의 Top-50에서 등장한 고유 상품 수 / 전체 상품 수
  ★ 검색 지표 (CLIP+FAISS):
      - MRR, NDCG@10, p95 latency

산출물:
  data/metrics.json  ← 대시보드가 즉시 사용
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score



# ──────────────────────────────────────────────
# 0. 경로
# ──────────────────────────────────────────────
DATA_DIR        = "data"
PRODUCTS_CSV    = f"{DATA_DIR}/products.csv"
USERS_CSV       = f"{DATA_DIR}/users.csv"
TRAIN_LOGS_CSV  = f"{DATA_DIR}/train_logs.csv"   # 인코더 fit 용
TEST_LOGS_CSV   = f"{DATA_DIR}/test_logs.csv"    # ⭐ 평가는 무조건 이 파일만
TEXT_INDEX_PATH = f"{DATA_DIR}/indices/text.index"
CAND_INDEX_PATH = f"{DATA_DIR}/indices/candidate_item.index"
TT_PATH         = f"{DATA_DIR}/models/two_tower.pth"
DEEPFM_PATH     = f"{DATA_DIR}/models/deepfm.pth"
METRICS_OUT     = f"{DATA_DIR}/metrics.json"

CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"

EVAL_SEARCH_N   = 500   # MRR/NDCG@10 평가 샘플 수
EVAL_RECO_N     = 500   # Recall@300 / HitRate@50 평가 샘플 수
COVERAGE_USERS  = 200   # Coverage 계산용 유저 수

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📐 Device: {device}")


# ══════════════════════════════════════════════════════════════════════
# 1.  검색 지표 (CLIP + FAISS)
# ══════════════════════════════════════════════════════════════════════
def evaluate_search() -> dict:
    print("\n🔍 [Search] MRR / NDCG@10 측정...")
    if not os.path.exists(TEXT_INDEX_PATH):
        print(f"  ⚠️  {TEXT_INDEX_PATH} 없음")
        return {"mrr": 0.0, "ndcg_10": 0.0, "latency_ms": 9999.0}

    df_prods = pd.read_csv(PRODUCTS_CSV)
    df_test  = pd.read_csv(TEST_LOGS_CSV)

    # FAISS row index = products.csv 순서
    pid_to_row = {pid: i for i, pid in enumerate(df_prods["product_id"].values)}

    # 평가셋: test의 purchase 이벤트 + 상품명 (쿼리)
    df_p = df_test[df_test["event_type"] == "purchase"]
    df_p = df_p[df_p["product_id"].isin(pid_to_row)]
    df_p = df_p.merge(df_prods[["product_id", "product_name"]], on="product_id", how="left")
    df_p = df_p.dropna(subset=["product_name"])

    n = min(EVAL_SEARCH_N, len(df_p))
    df_eval = df_p.sample(n, random_state=42)
    print(f"  평가 샘플: {n}건")

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip      = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    text_idx  = faiss.read_index(TEXT_INDEX_PATH)

    mrr_list, ndcg_list, lat_list = [], [], []

    with torch.no_grad():
        for _, row in tqdm(df_eval.iterrows(), total=n, desc="  Search"):
            query = str(row["product_name"])
            target_row = pid_to_row[row["product_id"]]

            tok = processor.tokenizer(
                query, return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            t_emb = clip.get_text_features(
                input_ids=tok["input_ids"].to(device),
                attention_mask=tok["attention_mask"].to(device),
            )
            if not isinstance(t_emb, torch.Tensor):
                pooler = getattr(t_emb, "pooler_output", None)
                embeds = getattr(t_emb, "text_embeds", None)
                t_emb = pooler if pooler is not None else embeds
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            t_np = t_emb.cpu().numpy().astype("float32")

            t0 = time.perf_counter()
            _, retrieved = text_idx.search(t_np, 10)
            lat_list.append((time.perf_counter() - t0) * 1000)

            rank = None
            for r, idx in enumerate(retrieved[0]):
                if idx == target_row:
                    rank = r + 1
                    break

            mrr_list.append(1.0 / rank if rank else 0.0)
            ndcg_list.append(1.0 / np.log2(rank + 1) if rank else 0.0)

    return {
        "mrr":        float(np.mean(mrr_list)),
        "ndcg_10":    float(np.mean(ndcg_list)),
        "latency_ms": round(float(np.percentile(lat_list, 95)), 1),
    }


# ══════════════════════════════════════════════════════════════════════
# 2.  추천 지표 (Two-Tower retrieval + DeepFM ranking)
# ══════════════════════════════════════════════════════════════════════

# DeepFM 아키텍처 (학습 시와 동일해야 함)
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


# Two-Tower 아키텍처 (학습 시와 동일)
class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_prods, n_persona, n_cat1, n_cat2, n_cat3, n_tier, emb_dim=32, out_dim=64):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.persona_emb = nn.Embedding(n_persona, emb_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

        self.item_emb = nn.Embedding(n_prods, emb_dim)
        self.cat1_emb = nn.Embedding(n_cat1, emb_dim)
        self.cat2_emb = nn.Embedding(n_cat2, emb_dim)
        self.cat3_emb = nn.Embedding(n_cat3, emb_dim)
        self.tier_emb = nn.Embedding(n_tier, emb_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim * 5, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def encode_user(self, u, p):
        x = torch.cat([self.user_emb(u), self.persona_emb(p)], dim=-1)
        return F.normalize(self.user_mlp(x), p=2, dim=-1)

    def encode_item(self, i, c1, c2, c3, t):
        x = torch.cat([
            self.item_emb(i),
            self.cat1_emb(c1),
            self.cat2_emb(c2),
            self.cat3_emb(c3),
            self.tier_emb(t),
        ], dim=-1)
        return F.normalize(self.item_mlp(x), p=2, dim=-1)



def evaluate_recommend() -> dict:
    print("\n🎯 [Recommend] Recall@300 / HitRate@50 / NDCG@50 / AUC / Coverage 측정...")

    df_users = pd.read_csv(USERS_CSV)
    df_prods = pd.read_csv(PRODUCTS_CSV)
    df_train = pd.read_csv(TRAIN_LOGS_CSV)
    df_test  = pd.read_csv(TEST_LOGS_CSV)

    # ─── 인코더 복원 (DeepFM 용) ───
    SPARSE = [
    "user_id",
    "product_id",
    "persona",
    "category_L1",
    "category_L2",
    "category_L3",
    "price_tier"
]

    train_full = df_train.merge(df_users, on="user_id").merge(df_prods, on="product_id")
    enc = {}
    for f in SPARSE:
        le = LabelEncoder()
        train_full[f] = le.fit_transform(train_full[f].astype(str))
        enc[f] = le
    feat_dims = {f: len(enc[f].classes_) for f in SPARSE}

    def encode_oov(df):
        X = np.zeros((len(df), len(SPARSE)), dtype=int)

        for i, f in enumerate(SPARSE):
            if f not in df.columns:
                df[f] = ""

            cls = enc[f].classes_
            v = df[f].astype(str).values
            m = np.isin(v, cls)

            if m.any():
                X[m, i] = enc[f].transform(v[m])

        return X


    # ─── DeepFM 로드 ───
    deepfm = DeepFM(feat_dims).to(device)
    if os.path.exists(DEEPFM_PATH):
        deepfm.load_state_dict(torch.load(DEEPFM_PATH, map_location=device))
        deepfm.eval()
        print("  ✅ DeepFM 로드")
    else:
        print(f"  ⚠️ {DEEPFM_PATH} 없음")
        return {k: 0.0 for k in ["recall_300", "hitrate_50", "ndcg_50", "auc", "coverage", "latency_ms"]}

    # ─── Two-Tower 로드 ───
    two_tower = None
    cand_idx  = None
    user_persona_map = {}  # user_id → persona_idx
    if os.path.exists(TT_PATH) and os.path.exists(CAND_INDEX_PATH):
        ckpt = torch.load(TT_PATH, map_location=device)
        two_tower = TwoTowerModel(
    ckpt["num_users"],
    ckpt["num_prods"],
    ckpt["num_persona"],
    ckpt["num_cat1"],
    ckpt["num_cat2"],
    ckpt["num_cat3"],
    ckpt["num_tier"],
).to(device)

        two_tower.load_state_dict(ckpt["state_dict"])
        two_tower.eval()

        cand_idx = faiss.read_index(CAND_INDEX_PATH)
        # 매핑 로드
        user_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_user_map.csv")
        user_persona_map = dict(zip(user_map["user_id"], user_map["persona_idx"]))
        user_idx_map     = dict(zip(user_map["user_id"], user_map["user_idx"]))
        prod_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_prod_map.csv")
        prod_idx_map     = dict(zip(prod_map["product_id"], prod_map["prod_idx"]))
        print("  ✅ Two-Tower + FAISS 로드")
    else:
        print(f"  ⚠️ Two-Tower/인덱스 없음 → Recall@300 = 0")
        user_idx_map = {}
        prod_idx_map = {}

    # ────────────────────────────────────────────────────
    # AUC 측정 (test 전체에서 view vs cart/purchase)
    # ────────────────────────────────────────────────────
    print("  [1/4] AUC 측정 중...")
    df_test_lbl = df_test[df_test["event_type"].isin(["view", "cart", "purchase"])].copy()
    df_test_lbl["label"] = df_test_lbl["event_type"].isin(["cart", "purchase"]).astype(float)
    df_test_full = df_test_lbl.merge(df_users, on="user_id").merge(df_prods, on="product_id")

    if len(df_test_full) > 30000:
        df_test_full = df_test_full.sample(30000, random_state=42)

    X_te = encode_oov(df_test_full)
    y_te = df_test_full["label"].values

    auc_val = 0.5
    if len(set(y_te)) >= 2:
        with torch.no_grad():
            preds = []
            bs = 4096
            for i in range(0, len(X_te), bs):
                xb = torch.tensor(X_te[i:i+bs], dtype=torch.long).to(device)
                preds.extend(deepfm(xb).cpu().numpy())
            auc_val = float(roc_auc_score(y_te, preds))
    print(f"     AUC = {auc_val:.4f}")

    # ────────────────────────────────────────────────────
    # Recall@300 / HitRate@50 / NDCG@50 (test의 purchase 기준)
    # ────────────────────────────────────────────────────
    print("  [2/4] Recall@300 / HitRate@50 / NDCG@50 측정 중...")
    df_pos = df_test[df_test["event_type"] == "purchase"].copy()
    df_pos = df_pos[df_pos["user_id"].isin(user_idx_map) & df_pos["product_id"].isin(prod_idx_map)]
    n_eval = min(EVAL_RECO_N, len(df_pos))
    df_eval = df_pos.sample(n_eval, random_state=42) if n_eval > 0 else df_pos

    # DeepFM 입력용 메타
    user_meta = df_users.set_index("user_id")["persona"].to_dict()
    prod_meta = df_prods.set_index("product_id")[[
    "category_L1",
    "category_L2",
    "category_L3",
    "price_tier"
]].to_dict("index")


    recall_hits, hit50_hits, ndcg50_sum = 0, 0, 0.0
    rec_lat = []

    if two_tower is not None and n_eval > 0:
        with torch.no_grad():
            for _, row in tqdm(df_eval.iterrows(), total=n_eval, desc="  Reco"):
                uid = row["user_id"]
                target_pid = row["product_id"]
                target_prod_idx = prod_idx_map[target_pid]

                # Stage 1: Two-Tower 후보 300개
                u_idx = torch.tensor([user_idx_map[uid]], dtype=torch.long).to(device)
                p_idx = torch.tensor([user_persona_map[uid]], dtype=torch.long).to(device)
                u_vec = two_tower.encode_user(u_idx, p_idx).cpu().numpy().astype("float32")
                faiss.normalize_L2(u_vec)

                t0 = time.perf_counter()
                _, cand_ids = cand_idx.search(u_vec, 300)  # (1, 300)  prod_idx 들
                cand_set = set(cand_ids[0].tolist())

                # Recall@300
                if target_prod_idx in cand_set:
                    recall_hits += 1

                # Stage 2: DeepFM으로 후보 300개 재정렬 → Top-50
                # prod_idx → product_id 역매핑
                prod_idx_to_id = {v: k for k, v in prod_idx_map.items()}
                cand_pids = [prod_idx_to_id[i] for i in cand_ids[0] if i in prod_idx_to_id]

                if not cand_pids:
                    continue

                # DeepFM 입력 구성
                cand_df = pd.DataFrame({"product_id": cand_pids})
                cand_df["user_id"] = uid
                cand_df["persona"] = user_meta.get(uid, "신중탐색")
                cand_df["category_L1"] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get("category_L1", ""))
                cand_df["category_L2"] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get("category_L2", ""))
                cand_df["category_L3"] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get("category_L3", ""))
                cand_df["price_tier"] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get("price_tier", "medium"))

                X_cand = encode_oov(cand_df)

                xb = torch.tensor(X_cand, dtype=torch.long).to(device)
                scores = deepfm(xb).cpu().numpy()
                rec_lat.append((time.perf_counter() - t0) * 1000)

                # Top-50 (점수 내림차순)
                if len(scores) <= 50:
                    top50_pids = cand_pids
                    rank_in_top = next((r+1 for r, p in enumerate(cand_pids) if p == target_pid), None)
                else:
                    top50_idx = np.argsort(-scores)[:50]
                    top50_pids = [cand_pids[i] for i in top50_idx]
                    rank_in_top = next((r+1 for r, p in enumerate(top50_pids) if p == target_pid), None)

                if rank_in_top is not None:
                    hit50_hits += 1
                    ndcg50_sum += 1.0 / np.log2(rank_in_top + 1)

        recall_300 = recall_hits / n_eval if n_eval else 0.0
        hitrate_50 = hit50_hits / n_eval if n_eval else 0.0
        ndcg_50    = ndcg50_sum / n_eval if n_eval else 0.0
        rec_lat_p95 = round(float(np.percentile(rec_lat, 95)), 1) if rec_lat else 9999.0
    else:
        recall_300, hitrate_50, ndcg_50, rec_lat_p95 = 0.0, 0.0, 0.0, 9999.0

    print(f"     Recall@300 = {recall_300:.4f}")
    print(f"     HitRate@50 = {hitrate_50:.4f}")
    print(f"     NDCG@50    = {ndcg_50:.4f}")
    print(f"     Lat p95    = {rec_lat_p95} ms")

    # ────────────────────────────────────────────────────
    # Coverage (샘플 유저들의 Top-50 합집합 / 전체 상품)
    # ────────────────────────────────────────────────────
    print("  [3/4] Coverage 측정 중...")
    recommended = set()

    if two_tower is not None:
        sample_uids = df_users.sample(min(COVERAGE_USERS, len(df_users)), random_state=42)["user_id"].tolist()
        sample_uids = [u for u in sample_uids if u in user_idx_map]
        prod_idx_to_id = {v: k for k, v in prod_idx_map.items()}

        with torch.no_grad():
            for uid in tqdm(sample_uids, desc="  Coverage"):
                u_idx = torch.tensor([user_idx_map[uid]], dtype=torch.long).to(device)
                p_idx = torch.tensor([user_persona_map[uid]], dtype=torch.long).to(device)
                u_vec = two_tower.encode_user(u_idx, p_idx).cpu().numpy().astype("float32")
                faiss.normalize_L2(u_vec)
                _, cand_ids = cand_idx.search(u_vec, 300)
                cand_pids = [prod_idx_to_id[i] for i in cand_ids[0] if i in prod_idx_to_id]
                if not cand_pids:
                    continue

                cand_df = pd.DataFrame({"product_id": cand_pids})
                cand_df["user_id"]  = uid
                cand_df["persona"]  = user_meta.get(uid, "신중탐색")
                cand_df["category_L1"] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get("category_L1", ""))
                cand_df["price_tier"]  = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get("price_tier", "medium"))
                X_cand = encode_oov(cand_df)
                xb = torch.tensor(X_cand, dtype=torch.long).to(device)
                scores = deepfm(xb).cpu().numpy()
                top50_idx = np.argsort(-scores)[:50]
                recommended.update(cand_df.iloc[top50_idx]["product_id"].tolist())

    coverage = len(recommended) / len(df_prods) if len(df_prods) > 0 else 0.0
    print(f"     Coverage   = {coverage:.4f}  ({len(recommended):,} / {len(df_prods):,})")

    return {
        "recall_300": float(recall_300),
        "hitrate_50": float(hitrate_50),
        "ndcg_50":    float(ndcg_50),
        "auc":        float(auc_val),
        "coverage":   float(coverage),
        "latency_ms": rec_lat_p95,
    }


# ══════════════════════════════════════════════════════════════════════
# 3.  메인
# ══════════════════════════════════════════════════════════════════════
def main():
    search_m = evaluate_search()
    reco_m   = evaluate_recommend()

    final = {
        "search": {
            "mrr":        round(search_m["mrr"],     4),
            "ndcg_10":    round(search_m["ndcg_10"], 4),
            "latency_ms": search_m["latency_ms"],
        },
        "recommend": {
            "recall_300": round(reco_m["recall_300"], 4),
            "auc":        round(reco_m["auc"],        4),
            "hitrate_50": round(reco_m["hitrate_50"], 4),
            "ndcg_50":    round(reco_m["ndcg_50"],    4),
            "coverage":   round(reco_m["coverage"],   4),
            "latency_ms": reco_m["latency_ms"],
        },
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("🔥 최종 지표")
    print(f"  MRR         {final['search']['mrr']:.4f}   (목표 ≥ 0.55)")
    print(f"  NDCG@10     {final['search']['ndcg_10']:.4f}   (목표 ≥ 0.50)")
    print(f"  Recall@300  {final['recommend']['recall_300']:.4f}   (목표 ≥ 0.30)")
    print(f"  AUC         {final['recommend']['auc']:.4f}   (목표 ≥ 0.70)")
    print(f"  HitRate@50  {final['recommend']['hitrate_50']:.4f}   (목표 ≥ 0.20)")
    print(f"  NDCG@50     {final['recommend']['ndcg_50']:.4f}   (목표 ≥ 0.08)")
    print(f"  Coverage    {final['recommend']['coverage']:.4f}   (목표 ≥ 0.20)")
    print("=" * 50)
    print(f"✅ {METRICS_OUT} 저장 완료")


if __name__ == "__main__":
    main()