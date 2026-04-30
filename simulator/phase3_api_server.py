import os
import io
import torch
import faiss
import pandas as pd
import numpy as np
import redis
import json
import random
import time
import re
from fastapi import FastAPI, UploadFile, File, Form, Query
from pydantic import BaseModel
from typing import Optional
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
from PIL import Image

# 우리가 Phase 2에서 만든 DeepFM 아키텍처 (가중치 로드를 위해 클래스 선언 필요)
import torch.nn as nn
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=16):
        super(DeepFM, self).__init__()
        self.sparse_features = list(feature_dims.keys())
        self.embeddings = nn.ModuleDict({feat: nn.Embedding(dim, embedding_dim) for feat, dim in feature_dims.items()})
        self.linear = nn.ModuleDict({feat: nn.Embedding(dim, 1) for feat, dim in feature_dims.items()})
        self.bias = nn.Parameter(torch.zeros(1))
        mlp_input_dim = len(feature_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )
    def forward(self, x):
        emb_list = [self.embeddings[feat](x[:, i]) for i, feat in enumerate(self.sparse_features)]
        linear_term = self.bias + sum([self.linear[feat](x[:, i]) for i, feat in enumerate(self.sparse_features)])
        emb_stack = torch.stack(emb_list, dim=1)
        sum_of_square = torch.sum(emb_stack, dim=1) ** 2
        square_of_sum = torch.sum(emb_stack ** 2, dim=1)
        fm_term = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1, keepdim=True)
        deep_input = torch.cat(emb_list, dim=1)
        deep_term = self.mlp(deep_input)
        return torch.sigmoid(linear_term + fm_term + deep_term).squeeze()

app = FastAPI(title="H&M Multi-Stage 추천 검색 API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# [추가] Redis 인메모리 DB 연결 세팅 (실시간 세션 저장용)
# ---------------------------------------------------------
try:
    redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ Redis 인메모리 DB 연결 성공!")
except Exception as e:
    print("⚠️ Redis가 켜져있지 않습니다! (추천 API 호출 시 에러가 날 수 있습니다)")

# --- 1. 글로벌 메모리 로드 (서버 켤 때 한 번만 실행) ---
print("🚀 [1/3] 멀티모달 CLIP & 벡터 DB 로딩 중...")
model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id).to(device)
clip_model.eval()

faiss_index = faiss.read_index("data/indices/text.index")

print("🚀 [2/3] 상품 및 유저 메타데이터 로딩 중...")
df_prods = pd.read_csv('data/products.csv')
df_users = pd.read_csv('data/users.csv')
df_logs = pd.read_csv('data/train_logs.csv') # 피처 차원 계산용

# DeepFM을 위한 인코더 복원 (간이 버전)
sparse_features = ['user_id', 'product_id', 'persona', 'category_L1', 'price_tier']
from sklearn.preprocessing import LabelEncoder
encoders = {}
temp_merged = df_logs.merge(df_users, on='user_id').merge(df_prods, on='product_id')
for feat in sparse_features:
    le = LabelEncoder()
    temp_merged[feat] = le.fit_transform(temp_merged[feat].astype(str))
    encoders[feat] = le

feature_dims = {feat: len(encoders[feat].classes_) for feat in sparse_features}

print("🚀 [3/3] DeepFM 랭킹 모델 로딩 중...")
deepfm_model = DeepFM(feature_dims).to(device)
deepfm_model.load_state_dict(torch.load('data/models/deepfm.pth', map_location=device))
deepfm_model.eval()

print("✅ 서버 준비 완료!")

# --- 2. API 엔드포인트 ---

@app.post("/api/search")
async def personalized_search(
    user_id: str = Form("U000058a12d"), 
    query: Optional[str] = Form(None),      
    file: Optional[UploadFile] = File(None) 
):
    """(기존) 멀티모달 기반의 검색 API"""
    if not query and not file:
        return {"error": "텍스트 검색어(query)나 이미지 파일(file) 중 하나는 반드시 입력해야 합니다."}

    text_emb = None
    image_emb = None
    search_type = "unknown"
    translated_query = None

    if query:
        translated_query = GoogleTranslator(source='auto', target='en').translate(query)
        text_input = translated_query
    else:
        text_input = "dummy"

    if file:
        image_data = await file.read()
        img_input = Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        img_input = Image.new('RGB', (224, 224), (0, 0, 0))

    inputs = clip_processor(text=[text_input], images=[img_input], return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        if query:
            text_emb = outputs.text_embeds
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        if file:
            image_emb = outputs.image_embeds
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)

    if query and file:
        search_type = "hybrid"
        final_emb = (text_emb + image_emb) / 2.0
        final_emb = final_emb / final_emb.norm(p=2, dim=-1, keepdim=True)
    elif query:
        search_type = "text"
        final_emb = text_emb
    else:
        search_type = "image"
        final_emb = image_emb

    final_emb_np = final_emb.cpu().numpy().astype('float32')
    distances, indices = faiss_index.search(final_emb_np, 100)
    candidate_ids = df_prods.iloc[indices[0]]['product_id'].values
    candidate_df = df_prods[df_prods['product_id'].isin(candidate_ids)].copy()

    user_info = df_users[df_users['user_id'] == user_id]
    if user_info.empty:
        return {
            "user_id": user_id,
            "persona": "신규가입자(Cold Start)", 
            "search_type": search_type,
            "original_query": query,
            "results": candidate_df.head(10)[['product_id', 'product_name', 'price']].to_dict('records')
        }
    
    candidate_df['user_id'] = user_id
    candidate_df['persona'] = user_info['persona'].values[0]
    
    X_pred = np.zeros((len(candidate_df), len(sparse_features)), dtype=int)
    for i, feat in enumerate(sparse_features):
        classes = encoders[feat].classes_
        vals = candidate_df[feat].astype(str).values
        X_pred[:, i] = np.where(np.isin(vals, classes), encoders[feat].transform(vals), 0)

    X_tensor = torch.tensor(X_pred, dtype=torch.long).to(device)
    
    with torch.no_grad():
        scores = deepfm_model(X_tensor).cpu().numpy()
        
    candidate_df['deepfm_score'] = scores
    final_results = candidate_df.sort_values(by='deepfm_score', ascending=False).head(10)
    
    return {
        "user_id": user_id,
        "persona": user_info['persona'].values[0],
        "search_type": search_type,
        "original_query": query,
        "results": final_results[['product_id', 'product_name', 'price', 'deepfm_score']].to_dict('records')
    }

# ---------------------------------------------------------
# [신규 API 1] 클릭 이벤트를 실시간으로 Redis에 저장 (Product ID 기반)
# ---------------------------------------------------------
@app.post("/api/click")
async def log_user_click(user_id: str = Form("U000058a12d"), item_id: str = Form(...)):
    """프론트엔드에서 유저가 클릭한 상품의 고유 ID(product_id)를 전송합니다."""
    session_key = f"session:{user_id}"
    
    existing_session = redis_client.get(session_key)
    clicked_items = json.loads(existing_session) if existing_session else []
    
    # 💡 이제 텍스트가 아니라 상품 ID(예: P0669715003)를 저장합니다!
    if item_id not in clicked_items:
        clicked_items.insert(0, item_id)
    clicked_items = clicked_items[:5] # 최근 클릭한 옷 5개만 기억
    
    redis_client.set(session_key, json.dumps(clicked_items), ex=3600)
    
    return {"message": f"상품 [{item_id}] 클릭이 Redis에 실시간 저장됨!", "current_session": clicked_items}

# ---------------------------------------------------------
# [신규 API 2] 실시간 추천 (Category 고정 + DeepFM 랭킹)
# ---------------------------------------------------------
@app.get("/api/recommend")
async def get_home_recommendations(
    user_id: str = Query("U000058a12d", description="추천을 받을 유저 ID"), 
    top_n: int = Query(10, description="반환할 추천 상품 개수")
):
    start_time = time.time()
    user_info = df_users[df_users['user_id'] == user_id]
    
    session_data = redis_client.get(f"session:{user_id}")
    recent_items = json.loads(session_data) if session_data else []

    if user_info.empty:
        fallback_items = df_prods.sample(top_n) 
        return {"user_id": user_id, "recommendations": fallback_items.to_dict('records')}
        
    user_persona = user_info['persona'].values[0]

    # 💡 [질문자님 아이디어 적용] 텍스트 검색을 버리고, DB의 메타데이터(Category)를 활용!
    if recent_items:
        # 1. 유저가 최근 클릭한 상품 ID들의 실제 DB 정보를 싹 다 가져옵니다.
        clicked_prods = df_prods[df_prods['product_id'].isin(recent_items)]
        
        if not clicked_prods.empty:
            # 2. 💡 [핵심] L1(대분류), L2(중분류), L3(소분류)의 '정확한 족보(조합)'를 중복 없이 추출합니다.
            # 예: [('Baby/Children', 'Children Sizes 134-170', 'Garment Upper body')]
            target_categories = clicked_prods[['category_L1', 'category_L2', 'category_L3']].drop_duplicates()
            
            # 3. 🔒 전체 상품 DB에서 이 "완벽하게 일치하는 카테고리 조합"을 가진 옷들만 교집합(inner join)으로 퍼옵니다!
            # 여성복을 봤으면 여성복 상의만, 아동복을 봤으면 아동복 상의만 정확히 매칭됩니다.
            candidate_pool = df_prods.merge(target_categories, on=['category_L1', 'category_L2', 'category_L3'], how='inner')
            sample_size = min(100, len(candidate_pool))
            candidate_df = candidate_pool.sample(sample_size).copy()
        else:
            candidate_df = df_prods.sample(100).copy()
    else:
        candidate_df = df_prods.sample(100).copy()

    # 부족하면 채우기
    if len(candidate_df) < 100:
        candidate_df = pd.concat([candidate_df, df_prods.sample(100 - len(candidate_df))]).copy()

    candidate_df['user_id'] = user_id
    candidate_df['persona'] = user_persona
    
    # 🧠 [AI 랭킹] 카테고리는 이미 고정되었으니, DeepFM은 유저 취향(페르소나)과 가격(price_tier)만 집중해서 점수를 매깁니다!
    X_pred = np.zeros((len(candidate_df), len(sparse_features)), dtype=int)
    for i, feat in enumerate(sparse_features):
        classes = encoders[feat].classes_
        vals = candidate_df[feat].astype(str).values
        X_pred[:, i] = np.where(np.isin(vals, classes), encoders[feat].transform(vals), 0)

    X_tensor = torch.tensor(X_pred, dtype=torch.long).to(device)
    with torch.no_grad():
        candidate_df['deepfm_score'] = deepfm_model(X_tensor).cpu().numpy()
        
    # MAB 비즈니스 로직 적용
    exploitation_count = top_n - 2
    best_items = candidate_df.sort_values(by='deepfm_score', ascending=False).head(exploitation_count)
    exploration_items = df_prods.sample(2) 

    final_recs = []
    for _, row in best_items.iterrows():
        final_recs.append({"product_id": row['product_id'], "score": float(row['deepfm_score']), "reason": "personalized_deepfm", "is_exploration": False})
    for _, row in exploration_items.iterrows():
        final_recs.append({"product_id": row['product_id'], "score": 0.0, "reason": "mab_exploration", "is_exploration": True})
        
    random.shuffle(final_recs)

    return {
        "user_id": user_id,
        "persona": user_persona,
        "pipeline_latency_ms": int((time.time() - start_time) * 1000),
        "session_context": {"recent_clicked_item_ids": recent_items},
        "recommendations": final_recs
    }
# ---------------------------------------------------------
# [신규 API 3] 무중단 배포 (새롭게 학습된 뇌 갈아끼우기)
# ---------------------------------------------------------
@app.post("/api/reload-model")
async def reload_model():
    try:
        # 하드디스크에 있는 최신 deepfm.pth 파일을 다시 읽어와서 덮어씌웁니다.
        deepfm_model.load_state_dict(torch.load('data/models/deepfm.pth', map_location=device))
        deepfm_model.eval()
        return {"message": "✅ 최신 트렌드를 반영한 새로운 AI 모델로 무중단 교체 완료!"}
    except Exception as e:
        return {"error": f"모델 로딩 실패: {str(e)}"}