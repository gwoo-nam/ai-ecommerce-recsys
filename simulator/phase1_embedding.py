import os
import torch
import pandas as pd
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1. 설정 및 디렉토리 준비
os.makedirs('data/indices', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. CLIP 모델 로드
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)
model.eval() # 평가 모드

# 3. 💡 [핵심 수정] 실제 H&M 이미지 로드 함수
def load_hm_image(product_id):
    """실제 H&M 데이터셋의 폴더 구조에서 이미지를 불러옵니다."""
    raw_id = product_id[1:]  # 'P0108775015' -> '0108775015' 추출
    folder_prefix = raw_id[:3]  # 앞 3자리가 하위 폴더명
    
    img_path = f"data/raw/images/{folder_prefix}/{raw_id}.jpg"
    
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    else:
        # 혹시 누락된 이미지가 있다면 에러 방지용 검은색 이미지 반환
        return Image.new('RGB', (224, 224), (0, 0, 0))

# 4. 임베딩 추출 및 FAISS 인덱싱 메인 로직
def build_multimodal_index():
    print("상품 메타데이터 로드 중...")
    df_prod = pd.read_csv('data/products.csv')
    
    # HNSW 인덱스 생성 (벡터 차원: 512)
    embedding_dim = 512 
    # M=32 (연결 노드 수), HNSW는 빠르고 정확도가 높아 멀티모달 검색에 적합합니다.
    text_index = faiss.IndexHNSWFlat(embedding_dim, 32)
    image_index = faiss.IndexHNSWFlat(embedding_dim, 32)
    
    text_embeddings = []
    image_embeddings = []
    
    # 배치 처리 (메모리 초과 방지)
    batch_size = 64
    
    print("CLIP 모델로 실제 H&M 텍스트 및 이미지 임베딩 추출 시작... (시간이 다소 소요됩니다)")
    with torch.no_grad():
        for i in tqdm(range(0, len(df_prod), batch_size)):
            batch_df = df_prod.iloc[i:i+batch_size]
            
            # 💡 [핵심 수정] 단순 카테고리가 아닌 디테일한 상품 설명 전체를 활용
            # 혹시 결측치가 있을 경우를 대비해 fillna("") 처리 추가
            texts = batch_df['product_name'].fillna("").tolist()
            
            # 💡 [핵심 수정] 진짜 옷 사진 로드
            images = [load_hm_image(row['product_id']) for _, row in batch_df.iterrows()]
            
            # 전처리 및 모델 통과 (텍스트가 너무 길면 잘리도록 truncation=True 추가)
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model(**inputs)
            
            # 텍스트 임베딩 추출 및 정규화
            text_emb = outputs.text_embeds
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            
            # 이미지 임베딩 추출 및 정규화
            image_emb = outputs.image_embeds
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            
            text_embeddings.append(text_emb.cpu().numpy())
            image_embeddings.append(image_emb.cpu().numpy())

    # 리스트를 하나의 Numpy 배열로 병합
    text_embeddings_np = np.vstack(text_embeddings).astype('float32')
    image_embeddings_np = np.vstack(image_embeddings).astype('float32')
    
    # FAISS 인덱스에 데이터 추가
    print("FAISS HNSW 인덱스 구축 중...")
    text_index.add(text_embeddings_np)
    image_index.add(image_embeddings_np)
    
    # 인덱스 디스크에 저장
    faiss.write_index(text_index, "data/indices/text.index")
    faiss.write_index(image_index, "data/indices/image.index")
    print("✅ 실제 데이터 기반 멀티모달 벡터 DB 구축 및 저장 완료! (data/indices/)")

if __name__ == "__main__":
    build_multimodal_index()