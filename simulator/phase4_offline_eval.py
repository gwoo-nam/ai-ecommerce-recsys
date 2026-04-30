"""
phase4_offline_eval.py
──────────────────────
실제 CLIP+FAISS / Two-Tower 모델을 이용해 오프라인 지표를 계산하고
결과를 data/search_metrics.json 으로 저장합니다.
이후 phase4_retrain_job.py 가 이 파일을 읽어 최종 metrics.json 을 완성합니다.

계산 지표
  - MRR         : CLIP 텍스트 검색에서 정답 상품의 역순위 평균
  - NDCG@10     : 상위 10개 결과에서의 NDCG
  - Recall@300  : Two-Tower FAISS 후보 300개 안에 정답 포함 비율
  - search_latency_ms : FAISS 검색 p95 레이턴시 (ms)
"""

import os
import time
import json
import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# 0.  경로 / 설정
# ──────────────────────────────────────────────
DATA_DIR          = "data"
PRODUCTS_CSV      = os.path.join(DATA_DIR, "products.csv")
USERS_CSV         = os.path.join(DATA_DIR, "users.csv")
TRAIN_LOGS_CSV    = os.path.join(DATA_DIR, "train_logs.csv")
TEXT_INDEX_PATH   = os.path.join(DATA_DIR, "indices", "text.index")
CAND_INDEX_PATH   = os.path.join(DATA_DIR, "indices", "candidate_item.index")
TWO_TOWER_PTH     = os.path.join(DATA_DIR, "models", "two_tower.pth")
OUT_JSON          = os.path.join(DATA_DIR, "search_metrics.json")

CLIP_MODEL_ID     = "openai/clip-vit-base-patch32"
EVAL_SAMPLE_N     = 500   # 평가에 쓸 purchase 로그 수 (속도/정밀도 균형)
RECALL_SAMPLE_N   = 300   # Recall@300 평가에 쓸 유저-상품 쌍 수
LATENCY_WARMUP    = 10    # 레이턴시 측정 warmup 횟수
LATENCY_REPEAT    = 100   # 레이턴시 측정 반복 횟수


# ──────────────────────────────────────────────
# 1.  MRR / NDCG@10  —  CLIP Text + FAISS
# ──────────────────────────────────────────────
def evaluate_search(device: str) -> dict:
    """
    구매(purchase) 로그에서 상품명을 텍스트 쿼리로 사용해
    FAISS 에서 검색했을 때 정답 상품이 몇 번째에 나타나는지 측정합니다.
    """
    print("\n🔍 [CLIP+FAISS] MRR / NDCG@10 측정 시작...")

    if not os.path.exists(TEXT_INDEX_PATH):
        print(f"  ⚠️  {TEXT_INDEX_PATH} 가 없습니다. CLIP 인덱싱(phase1_embedding.py)을 먼저 실행하세요.")
        return {"mrr": 0.0, "ndcg_10": 0.0, "latency_ms": 9999}

    # 데이터 로드
    df_prods = pd.read_csv(PRODUCTS_CSV)
    df_logs  = pd.read_csv(TRAIN_LOGS_CSV)

    # product_id → FAISS row index 매핑 (products.csv 순서가 인덱스 순서와 동일)
    prod_id_to_idx = {pid: i for i, pid in enumerate(df_prods["product_id"].values)}

    # 평가 셋: purchase 이벤트 중 FAISS 인덱스에 등록된 상품만 사용
    df_purchase = df_logs[df_logs["event_type"] == "purchase"].copy()
    df_purchase = df_purchase[df_purchase["product_id"].isin(prod_id_to_idx)]
    df_purchase = df_purchase.merge(
        df_prods[["product_id", "product_name"]], on="product_id", how="left"
    ).dropna(subset=["product_name"])

    n_sample = min(EVAL_SAMPLE_N, len(df_purchase))
    df_eval  = df_purchase.sample(n_sample, random_state=42)

    # CLIP 로드
    print("  CLIP 모델 로딩 중...")
    processor  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_model.eval()

    # FAISS 인덱스 로드
    text_index = faiss.read_index(TEXT_INDEX_PATH)

    mrr_list, ndcg_list, latency_list = [], [], []

    with torch.no_grad():
        for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="  MRR/NDCG"):
            query = str(row["product_name"])
            target_id = row["product_id"]
            target_idx = prod_id_to_idx[target_id]

            # 텍스트 임베딩
            # CLIPProcessor 는 text 전용으로 호출해야 pixel_values 가 포함되지 않음
            tok = processor.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            input_ids      = tok["input_ids"].to(device)
            attention_mask = tok["attention_mask"].to(device)

            # transformers 버전에 따라 get_text_features 반환 타입이 다름:
            #  - 4.x : torch.Tensor 직접 반환
            #  - 5.x : BaseModelOutputWithPooling 객체 반환 (pooler_output 에 임베딩)
            text_out = clip_model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if isinstance(text_out, torch.Tensor):
                text_emb = text_out
            elif hasattr(text_out, "pooler_output"):
                text_emb = text_out.pooler_output
            elif hasattr(text_out, "text_embeds"):
                text_emb = text_out.text_embeds
            else:
                # 마지막 안전장치: 첫 번째 Tensor 속성을 찾기
                text_emb = text_out[0] if isinstance(text_out, tuple) else text_out

            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb_np = text_emb.cpu().numpy().astype("float32")

            # 레이턴시도 여기서 측정 (단건 기준)
            t0 = time.perf_counter()
            _, retrieved_indices = text_index.search(text_emb_np, 10)
            latency_list.append((time.perf_counter() - t0) * 1000)

            # 정답 순위 계산
            rank = None
            for r, idx in enumerate(retrieved_indices[0]):
                if idx == target_idx:
                    rank = r + 1
                    break

            mrr_list.append(1.0 / rank if rank else 0.0)
            # NDCG@10: ideal DCG = 1 (단일 정답)
            ndcg_list.append(1.0 / np.log2(rank + 1) if rank else 0.0)

    # p95 레이턴시
    latency_p95 = float(np.percentile(latency_list, 95)) if latency_list else 9999.0

    return {
        "mrr":        float(np.mean(mrr_list)),
        "ndcg_10":    float(np.mean(ndcg_list)),
        "latency_ms": round(latency_p95, 1),
    }


# ──────────────────────────────────────────────
# 2.  Recall@300  —  Two-Tower + FAISS
# ──────────────────────────────────────────────
def evaluate_two_tower() -> dict:
    """
    Two-Tower 의 User 임베딩과 FAISS candidate_item.index 를 이용해
    후보 300개 안에 실제 구매 상품이 포함되는 비율(Recall@300)을 계산합니다.
    """
    print("\n🗼 [Two-Tower+FAISS] Recall@300 측정 시작...")

    if not os.path.exists(CAND_INDEX_PATH) or not os.path.exists(TWO_TOWER_PTH):
        print(f"  ⚠️  candidate_item.index 또는 two_tower.pth 가 없습니다.")
        return {"recall_300": 0.0}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_users = pd.read_csv(USERS_CSV)
    df_prods = pd.read_csv(PRODUCTS_CSV)
    df_logs  = pd.read_csv(TRAIN_LOGS_CSV)

    # 인코더 복원 (two_tower.py 와 동일 순서)
    user_encoder = LabelEncoder()
    prod_encoder = LabelEncoder()
    user_encoder.fit(df_users["user_id"].values)
    prod_encoder.fit(df_prods["product_id"].values)

    prod_id_to_encidx = {pid: i for i, pid in enumerate(prod_encoder.classes_)}
    user_id_to_encidx = {uid: i for i, uid in enumerate(user_encoder.classes_)}

    # 평가 셋
    df_purchase = df_logs[df_logs["event_type"] == "purchase"].copy()
    df_purchase = df_purchase[
        df_purchase["product_id"].isin(prod_id_to_encidx) &
        df_purchase["user_id"].isin(user_id_to_encidx)
    ]
    n_sample = min(RECALL_SAMPLE_N, len(df_purchase))
    df_eval  = df_purchase.sample(n_sample, random_state=42)

    # Two-Tower 모델 로드
    import torch.nn as nn
    class TwoTowerModel(nn.Module):
        def __init__(self, n_users, n_items, emb_dim=64):
            super().__init__()
            self.user_embedding = nn.Embedding(n_users, emb_dim)
            self.item_embedding = nn.Embedding(n_items, emb_dim)
        def forward(self, u, i):
            return (self.user_embedding(u) * self.item_embedding(i)).sum(-1).sigmoid()

    n_users = len(user_encoder.classes_)
    n_prods = len(prod_encoder.classes_)
    tt_model = TwoTowerModel(n_users, n_prods, emb_dim=64).to(device)
    tt_model.load_state_dict(torch.load(TWO_TOWER_PTH, map_location=device))
    tt_model.eval()

    # FAISS 후보 인덱스 로드
    cand_index = faiss.read_index(CAND_INDEX_PATH)

    hits = 0
    with torch.no_grad():
        for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="  Recall@300"):
            u_idx = user_id_to_encidx[row["user_id"]]
            target_enc_idx = prod_id_to_encidx[row["product_id"]]

            # User 임베딩 추출
            u_tensor = torch.tensor([u_idx], dtype=torch.long).to(device)
            u_emb = tt_model.user_embedding(u_tensor).cpu().numpy().astype("float32")
            faiss.normalize_L2(u_emb)

            # FAISS 검색
            _, indices = cand_index.search(u_emb, 300)
            if target_enc_idx in indices[0]:
                hits += 1

    recall = hits / n_sample if n_sample > 0 else 0.0
    return {"recall_300": float(recall)}


# ──────────────────────────────────────────────
# 3.  메인 실행
# ──────────────────────────────────────────────
def run_full_offline_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📐 [Offline Eval] 디바이스: {device}")

    search_metrics   = evaluate_search(device)
    retrieval_metrics = evaluate_two_tower()

    results = {**search_metrics, **retrieval_metrics}

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 40)
    print("🔥 [Offline Eval 최종 결과]")
    print(f"  MRR          : {results['mrr']:.4f}   (목표 >= 0.55)")
    print(f"  NDCG@10      : {results['ndcg_10']:.4f}   (목표 >= 0.50)")
    print(f"  Recall@300   : {results['recall_300']:.4f}   (목표 >= 0.30)")
    print(f"  Search p95   : {results['latency_ms']} ms  (목표 <= 200ms)")
    print("=" * 40)
    print(f"✅ 결과가 {OUT_JSON} 에 저장되었습니다.")
    return results


if __name__ == "__main__":
    run_full_offline_evaluation()