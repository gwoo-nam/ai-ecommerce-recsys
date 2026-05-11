"""
phase3_api_server.py  (전면 개편판: 진짜 Multi-Stage 추천)
─────────────────────────────────────────────────────────
명세서 부합 사항:

  Stage 1 │ Candidate Generation (Two-Tower + FAISS)
            - User Tower(persona 포함) → FAISS IP search → 후보 300개
  Stage 2 │ Ranking (DeepFM)
            - User/Item/Cross/Context 피처
  Stage 3 │ Re-ranking (비즈니스 로직 + MAB)
            - 동일 카테고리 연속 3개 금지 (다양성)
            - 신규 유저(이력 5개 미만): 인기 + 트렌딩으로 폴백
            - 신규 상품(등록 7일 이내): 노출 부스팅
            - Epsilon-Greedy MAB: 상위 N개 중 1~2개를 탐색 슬롯으로

  + Session Encoder (간이 GRU): 최근 클릭 시퀀스 → 단기 관심 임베딩
                                 장기 선호(User Tower) ⊕ 단기 관심 결합
  + 응답 필수 필드: search_type/results/latency_ms/total_count
                   user_id/recommendations/pipeline_latency/session_context
"""
import os
import io
import json
import time
import random
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import redis
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, Query
from pydantic import BaseModel
from typing import Optional, Literal
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_DIR        = "data"
TEXT_INDEX_PATH = f"{DATA_DIR}/indices/text.index"
CAND_INDEX_PATH = f"{DATA_DIR}/indices/candidate_item.index"
DEEPFM_PATH     = f"{DATA_DIR}/models/deepfm.pth"
TT_PATH         = f"{DATA_DIR}/models/two_tower.pth"

CANDIDATE_K     = 300       # Stage 1 후보 개수
RANK_TOP_N      = 50        # Stage 2 → Top-N 통과
SESSION_LEN     = 10        # Redis 세션 시퀀스 길이
NEW_PRODUCT_DAYS = 7        # 신규 상품 정의: 7일 이내
NEW_USER_HISTORY_THRESHOLD = 5  # 신규 유저 정의: 행동 5개 미만
EPSILON         = 0.1       # MAB exploration ratio
MAX_SAME_CATEGORY_RUN = 2   # 연속 3개 금지 → 같은 카테고리 연속 2개까지 허용

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Multi-Stage Recommender API")


# ──────────────────────────────────────────────
# 모델 정의 (학습 코드와 정확히 동일해야 함)
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


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_prods, n_persona, n_cat1, n_cat2, n_cat3, n_tier, emb_dim=32, out_dim=64):
        super().__init__()
        self.user_emb    = nn.Embedding(n_users,   emb_dim)
        self.persona_emb = nn.Embedding(n_persona, emb_dim)
        self.user_mlp = nn.Sequential(nn.Linear(emb_dim * 2, 128), nn.ReLU(), nn.Linear(128, out_dim))
        
        self.item_emb = nn.Embedding(n_prods, emb_dim)
        self.cat1_emb = nn.Embedding(n_cat1, emb_dim)
        self.cat2_emb = nn.Embedding(n_cat2, emb_dim)
        self.cat3_emb = nn.Embedding(n_cat3, emb_dim)
        self.tier_emb = nn.Embedding(n_tier, emb_dim)
        self.item_mlp = nn.Sequential(nn.Linear(emb_dim * 5, 128), nn.ReLU(), nn.Linear(128, out_dim))
        self.out_dim  = out_dim

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


class SessionGRU(nn.Module):
    """간이 세션 인코더: 최근 클릭한 item embedding 시퀀스를 GRU로 요약."""
    def __init__(self, item_emb_layer: nn.Embedding, hidden=64, out_dim=64):
        super().__init__()
        self.item_emb_layer = item_emb_layer
        self.gru = nn.GRU(item_emb_layer.embedding_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, item_idx_seq: torch.Tensor):
        # (1, T) → (1, T, emb) → (1, hidden)
        emb = self.item_emb_layer(item_idx_seq)
        _, h = self.gru(emb)
        return F.normalize(self.proj(h.squeeze(0)), p=2, dim=-1)


# ──────────────────────────────────────────────
# Redis
# ──────────────────────────────────────────────
try:
    redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ Redis 연결 성공")
except Exception:
    try:
        redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Redis 연결 성공 (localhost)")
    except Exception as e:
        print(f"⚠️ Redis 연결 실패: {e}")
        redis_client = None


# ──────────────────────────────────────────────
# 글로벌 자원 로드
# ──────────────────────────────────────────────
print("🚀 [1/5] CLIP & 텍스트 FAISS 로드...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
text_faiss     = faiss.read_index(TEXT_INDEX_PATH)

print("🚀 [2/5] 메타데이터 로드...")
df_prods = pd.read_csv(f"{DATA_DIR}/products.csv")
df_users = pd.read_csv(f"{DATA_DIR}/users.csv")
df_logs  = pd.read_csv(f"{DATA_DIR}/train_logs.csv")

# 인기 상품 (purchase 횟수) — 신규유저/MAB 폴백용
popular_pids = (
    df_logs[df_logs["event_type"] == "purchase"]["product_id"]
    .value_counts().head(200).index.tolist()
)

# 트렌딩 상품 (최근 30% 시점의 view)
df_logs_dt = df_logs.copy()
try:
    df_logs_dt["timestamp"] = pd.to_datetime(df_logs_dt["timestamp"])
    cutoff = df_logs_dt["timestamp"].quantile(0.7)
    trending_pids = (
        df_logs_dt[(df_logs_dt["timestamp"] >= cutoff) & (df_logs_dt["event_type"] == "view")]["product_id"]
        .value_counts().head(200).index.tolist()
    )
except Exception:
    trending_pids = popular_pids[:]

# 신규 상품 (등록 7일 이내) — train_logs 마지막 날짜 기준
try:
    latest_date = df_logs_dt["timestamp"].max()
    first_seen = df_logs_dt.groupby("product_id")["timestamp"].min()
    new_pids = first_seen[first_seen >= latest_date - pd.Timedelta(days=NEW_PRODUCT_DAYS)].index.tolist()
except Exception:
    new_pids = []

print(f"  인기 상품 {len(popular_pids)} / 트렌딩 {len(trending_pids)} / 신규 {len(new_pids)}")

print("🚀 [3/5] DeepFM 인코더/모델 로드...")
SPARSE_FEATURES = ['user_id', 'product_id', 'persona', 'category_L1', 'category_L2', 'category_L3', 'price_tier']
encoders = {}
temp = df_logs.merge(df_users, on="user_id").merge(df_prods, on="product_id")
for f in SPARSE_FEATURES:
    le = LabelEncoder()
    temp[f] = le.fit_transform(temp[f].astype(str))
    encoders[f] = le
feature_dims = {f: len(encoders[f].classes_) for f in SPARSE_FEATURES}

deepfm = DeepFM(feature_dims).to(device)
deepfm.load_state_dict(torch.load(DEEPFM_PATH, map_location=device))
deepfm.eval()

print("🚀 [4/5] Two-Tower 로드...")
two_tower = None
candidate_faiss = None
user_idx_map = {}
user_persona_map = {}
prod_idx_map = {}
prod_idx_to_id = {}

if os.path.exists(TT_PATH) and os.path.exists(CAND_INDEX_PATH):
    ckpt = torch.load(TT_PATH, map_location=device)
    
    # 여기서 카테고리 1, 2, 3을 각각 불러오도록 수정됨!
    two_tower = TwoTowerModel(
        n_users=ckpt["num_users"], 
        n_prods=ckpt["num_prods"],
        n_persona=ckpt["num_persona"], 
        n_cat1=ckpt["num_cat1"], 
        n_cat2=ckpt["num_cat2"], 
        n_cat3=ckpt["num_cat3"], 
        n_tier=ckpt["num_tier"],
    ).to(device)
    
    two_tower.load_state_dict(ckpt["state_dict"])
    two_tower.eval()
    candidate_faiss = faiss.read_index(CAND_INDEX_PATH)

    user_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_user_map.csv")
    user_idx_map     = dict(zip(user_map["user_id"], user_map["user_idx"]))
    user_persona_map = dict(zip(user_map["user_id"], user_map["persona_idx"]))
    prod_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_prod_map.csv")
    prod_idx_map     = dict(zip(prod_map["product_id"], prod_map["prod_idx"]))
    prod_idx_to_id   = {v: k for k, v in prod_idx_map.items()}
    print("  ✅ Two-Tower + FAISS IndexIDMap 로드")

print("🚀 [5/5] 세션 GRU 초기화...")
session_encoder = None
if two_tower is not None:
    session_encoder = SessionGRU(two_tower.item_emb).to(device).eval()

# 빠른 조회용
prod_meta_dict = df_prods.set_index("product_id").to_dict("index")
user_persona_str = df_users.set_index("user_id")["persona"].to_dict()
all_pids_set = set(df_prods["product_id"].values)

print("✅ 서버 준비 완료!")


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def encode_for_deepfm(df: pd.DataFrame) -> np.ndarray:
    X = np.zeros((len(df), len(SPARSE_FEATURES)), dtype=int)
    for i, f in enumerate(SPARSE_FEATURES):
        cls = encoders[f].classes_
        v = df[f].astype(str).values
        m = np.isin(v, cls)
        X[m, i] = encoders[f].transform(v[m])
    return X


def get_session_recent_pids(user_id: str) -> list:
    if redis_client is None:
        return []
    raw = redis_client.get(f"session:{user_id}")
    return json.loads(raw) if raw else []


def diversity_rerank(ranked_pids: list, max_run: int = MAX_SAME_CATEGORY_RUN) -> list:
    """
    동일 카테고리 연속 3개 이상 금지 (= 같은 카테고리 연속 2개까지 허용).
    탐욕적으로 다음 후보를 골라 카테고리 run 제약을 만족시킴.
    """
    out, run_cat, run_count = [], None, 0
    pool = list(ranked_pids)

    while pool and len(out) < len(ranked_pids):
        picked_idx = None
        for i, pid in enumerate(pool):
            cat = prod_meta_dict.get(pid, {}).get("category_L1")
            if cat == run_cat and run_count >= max_run:
                continue
            picked_idx = i
            break

        if picked_idx is None:
            picked_idx = 0  # 제약 풀고 그냥 픽

        pid = pool.pop(picked_idx)
        cat = prod_meta_dict.get(pid, {}).get("category_L1")
        if cat == run_cat:
            run_count += 1
        else:
            run_cat, run_count = cat, 1
        out.append(pid)

    return out


# ──────────────────────────────────────────────
# Pydantic
# ──────────────────────────────────────────────
class LogEvent(BaseModel):
    user_id: str
    product_id: str
    event_type: Literal["search", "view", "cart", "purchase"]
    timestamp: int


# ══════════════════════════════════════════════
# /api/search  (멀티모달 검색, 명세 필수 필드 준수)
# ══════════════════════════════════════════════
@app.post("/api/search")
async def personalized_search(
    user_id: str = Form("U000058a12d"),
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
):
    t0 = time.perf_counter()
    if not query and not file:
        return {"error": "query 또는 file 중 하나는 필수입니다."}

    # 임베딩 생성
    text_emb, image_emb = None, None

    if query:
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(query)
        except Exception:
            translated = query
        tok = clip_processor.tokenizer(translated, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            tf = clip_model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            if not isinstance(tf, torch.Tensor):
                pooler = getattr(tf, "pooler_output", None)
                embeds = getattr(tf, "text_embeds", None)
                tf = pooler if pooler is not None else embeds
            text_emb = tf / tf.norm(dim=-1, keepdim=True)

    if file:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        proc = clip_processor(images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            img_f = clip_model.get_image_features(pixel_values=proc["pixel_values"])
            image_emb = img_f / img_f.norm(dim=-1, keepdim=True)

    # 검색 타입 결정
    if query and file:
        search_type = "hybrid"
        final_emb = (text_emb + image_emb) / 2.0
        final_emb = final_emb / final_emb.norm(dim=-1, keepdim=True)
    elif query:
        search_type = "text"
        final_emb = text_emb
    else:
        search_type = "image"
        final_emb = image_emb

    # FAISS 검색
    final_np = final_emb.cpu().numpy().astype("float32")
    distances, indices = text_faiss.search(final_np, max(top_k * 5, 50))
    candidate_ids = df_prods.iloc[indices[0]]["product_id"].values
    candidate_df = df_prods[df_prods["product_id"].isin(candidate_ids)].copy()

    # DeepFM 개인화 랭킹
    user_info = df_users[df_users["user_id"] == user_id]
    if user_info.empty:
        results_df = candidate_df.head(top_k)
        results_df = results_df.assign(score=1.0)
        persona = "신규가입자"
    else:
        persona = user_info["persona"].values[0]
        candidate_df["user_id"] = user_id
        candidate_df["persona"] = persona
        X_p = encode_for_deepfm(candidate_df)
        with torch.no_grad():
            scores = deepfm(torch.tensor(X_p, dtype=torch.long).to(device)).cpu().numpy()
        candidate_df["score"] = scores
        results_df = candidate_df.sort_values("score", ascending=False).head(top_k)

    latency = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "search_type": search_type,
        "results": [
            {
                "product_id": r["product_id"],
                "name":       r["product_name"],
                "score":      float(r["score"]),
                "price":      int(r["price"]),
            }
            for _, r in results_df.iterrows()
        ],
        "latency_ms": latency,
        "total_count": int(len(candidate_df)),
        "user_id": user_id,
        "persona": persona,
        "original_query": query,
    }


# ══════════════════════════════════════════════
# /api/log  (세션 추적)
# ══════════════════════════════════════════════
@app.post("/api/log")
async def receive_log(log: LogEvent):
    if redis_client is None:
        return {"message": "redis 미연결", "event": log.event_type}

    key = f"session:{log.user_id}"
    raw = redis_client.get(key)
    seq = json.loads(raw) if raw else []

    if log.event_type in ["search", "view", "cart", "purchase"]:
        if log.product_id in seq:
            seq.remove(log.product_id)
        seq.insert(0, log.product_id)
        seq = seq[:SESSION_LEN]
        redis_client.set(key, json.dumps(seq), ex=3600)

    # 클릭 횟수 카운트 (실시간 피처)
    redis_client.incr(f"click_count:{log.user_id}")
    redis_client.expire(f"click_count:{log.user_id}", 3600)

    return {"message": "ok", "event": log.event_type}


# ══════════════════════════════════════════════
# /api/recommend  (진짜 Multi-Stage)
# ══════════════════════════════════════════════
@app.get("/api/recommend")
async def recommend(
    user_id: str = Query("U000058a12d"),
    top_n: int = Query(10),
):
    t_total = time.perf_counter()

    # 컨텍스트 수집
    user_info  = df_users[df_users["user_id"] == user_id]
    persona    = user_info["persona"].values[0] if not user_info.empty else None
    recent_pids = get_session_recent_pids(user_id)
    history_count = len(recent_pids)

    # 신규 유저 폴백
    is_new_user = (persona is None) or (history_count < NEW_USER_HISTORY_THRESHOLD and user_info.empty)
    # 명세에 따라 "행동 이력 5개 미만"을 폴백 조건으로 쓴다
    cold_start = history_count < NEW_USER_HISTORY_THRESHOLD

    session_interest = None
    if recent_pids:
        cats = [prod_meta_dict.get(p, {}).get("category_L1") for p in recent_pids]
        cats = [c for c in cats if c]
        if cats:
            session_interest = max(set(cats), key=cats.count)

    # ──────── Stage 1: Candidate Generation ────────
    t_s1 = time.perf_counter()
    candidate_pids = []

    can_use_two_tower = (two_tower is not None) and (user_id in user_idx_map)
    if can_use_two_tower:
        u_idx = torch.tensor([user_idx_map[user_id]], dtype=torch.long).to(device)
        p_idx = torch.tensor([user_persona_map[user_id]], dtype=torch.long).to(device)

        with torch.no_grad():
            u_vec = two_tower.encode_user(u_idx, p_idx)  # (1, D)

            # 단기 관심(세션) ⊕ 장기 선호(User Tower)
            if session_encoder is not None and recent_pids:
                seq_idx = [prod_idx_map[p] for p in recent_pids if p in prod_idx_map]
                if seq_idx:
                    seq_t = torch.tensor([seq_idx], dtype=torch.long).to(device)
                    s_vec = session_encoder(seq_t)  # (1, D)
                    u_vec = F.normalize(0.6 * u_vec + 0.4 * s_vec, p=2, dim=-1)

        u_np = u_vec.cpu().numpy().astype("float32")
        faiss.normalize_L2(u_np)
        _, cand_ids = candidate_faiss.search(u_np, CANDIDATE_K)
        candidate_pids = [prod_idx_to_id[i] for i in cand_ids[0] if i in prod_idx_to_id]

    # 콜드스타트 또는 Two-Tower 사용 불가 시: 인기 + 트렌딩으로 폴백
    if cold_start or not candidate_pids:
        fallback = list(dict.fromkeys(popular_pids + trending_pids))[:CANDIDATE_K]
        candidate_pids = fallback if not candidate_pids else candidate_pids
        # 부족하면 채우기
        if len(candidate_pids) < CANDIDATE_K:
            for p in fallback:
                if p not in candidate_pids:
                    candidate_pids.append(p)
                if len(candidate_pids) >= CANDIDATE_K:
                    break

    candidate_ms = round((time.perf_counter() - t_s1) * 1000, 1)

    # ──────── Stage 2: Ranking (DeepFM) ────────
    t_s2 = time.perf_counter()
    cand_df = pd.DataFrame({"product_id": candidate_pids})
    cand_df["user_id"] = user_id
    cand_df["persona"] = persona if persona else "신중탐색"
    cand_df = cand_df.merge(
        df_prods[["product_id", "category_L1", "price_tier"]],
        on="product_id", how="left"
    )
    cand_df["category_L1"] = cand_df["category_L1"].fillna("Ladieswear")
    cand_df["price_tier"]  = cand_df["price_tier"].fillna("medium")

    X_p = encode_for_deepfm(cand_df)
    with torch.no_grad():
        ranking_scores = deepfm(torch.tensor(X_p, dtype=torch.long).to(device)).cpu().numpy()
    cand_df["score"] = ranking_scores

    ranked = cand_df.sort_values("score", ascending=False).head(RANK_TOP_N)
    ranking_ms = round((time.perf_counter() - t_s2) * 1000, 1)

    # ──────── Stage 3: Re-ranking (다양성 + 신규상품 + MAB) ────────
    t_s3 = time.perf_counter()

    ranked_pids = ranked["product_id"].tolist()
    score_map = dict(zip(ranked["product_id"], ranked["score"]))

    # 1) 다양성 (동일 카테고리 연속 3개 금지)
    diverse_pids = diversity_rerank(ranked_pids)

    # 2) 신규 상품 노출 부스팅: 후보에서 신규상품 1개를 상위 3 안에 강제 삽입
    new_in_candidates = [p for p in diverse_pids if p in set(new_pids)]
    if new_in_candidates and not cold_start:
        boost = new_in_candidates[0]
        diverse_pids.remove(boost)
        diverse_pids.insert(min(2, len(diverse_pids)), boost)

    # 3) Top-N 자르기 + MAB 탐색 슬롯 (Epsilon-Greedy)
    final_pids = diverse_pids[:max(top_n - 2, 1)]  # exploitation
    explore_count = max(1, top_n - len(final_pids))  # 1~2개 탐색
    explore_pool = [p for p in popular_pids + trending_pids if p not in final_pids]
    random.shuffle(explore_pool)
    explore_picks = explore_pool[:explore_count]

    # 결과 조립
    final_recs = []
    for pid in final_pids:
        reason = "personalized_deepfm"
        if pid in set(new_pids):
            reason = "new_product_boost"
        elif session_interest and prod_meta_dict.get(pid, {}).get("category_L1") == session_interest:
            reason = "session_interest"
        final_recs.append({
            "product_id": pid,
            "score": float(score_map.get(pid, 0.0)),
            "reason": reason,
            "is_exploration": False,
        })

    for pid in explore_picks:
        final_recs.append({
            "product_id": pid,
            "score": 0.0,
            "reason": "mab_exploration",
            "is_exploration": True,
        })

    # 셔플하지 말고 다양성 정렬 한 번 더
    final_recs = final_recs[:top_n]

    reranking_ms = round((time.perf_counter() - t_s3) * 1000, 1)
    total_ms = round((time.perf_counter() - t_total) * 1000, 1)

    return {
        "user_id": user_id,
        "persona": persona,
        "recommendations": final_recs,
        "pipeline_latency": {
            "candidate_ms":  candidate_ms,
            "ranking_ms":    ranking_ms,
            "reranking_ms":  reranking_ms,
            "total_ms":      total_ms,
        },
        "session_context": {
            "recent_clicks":     recent_pids,
            "recent_clicked_item_ids": recent_pids,  # 호환
            "session_interest":  session_interest,
            "history_count":     history_count,
            "is_cold_start":     cold_start,
        },
    }


# ══════════════════════════════════════════════
# /api/reload-model  (CT 후 무중단 갱신)
# ══════════════════════════════════════════════
@app.post("/api/reload-model")
async def reload_model():
    try:
        deepfm.load_state_dict(torch.load(DEEPFM_PATH, map_location=device))
        deepfm.eval()
        return {"message": "✅ DeepFM 가중치 갱신 완료"}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════
# /api/health  (간단 헬스 체크)
# ══════════════════════════════════════════════
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "two_tower_loaded": two_tower is not None,
        "deepfm_loaded": deepfm is not None,
        "redis_connected": redis_client is not None,
        "n_products": len(df_prods),
        "n_users": len(df_users),
    }