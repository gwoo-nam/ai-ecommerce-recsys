import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 데이터 로드 및 전처리 (Label Encoding)
print("데이터 로딩 중...")
df_users = pd.read_csv('data/users.csv')
df_prods = pd.read_csv('data/products.csv')
df_logs = pd.read_csv('data/train_logs.csv')

# 빠른 테스트를 위해 데이터 사이즈를 줄이려면 아래 주석을 푸세요
# df_logs = df_logs.head(50000) 

user_encoder = LabelEncoder()
prod_encoder = LabelEncoder()

df_users['user_idx'] = user_encoder.fit_transform(df_users['user_id'])
df_prods['prod_idx'] = prod_encoder.fit_transform(df_prods['product_id'])

# 로그 데이터에 인덱스 매핑 (조인)
df_logs = df_logs.merge(df_users[['user_id', 'user_idx']], on='user_id')
df_logs = df_logs.merge(df_prods[['product_id', 'prod_idx']], on='product_id')

num_users = len(user_encoder.classes_)
num_prods = len(prod_encoder.classes_)

# 2. Dataset 구성 (명세서 요구사항: 1:4 Negative Sampling 적용)
class TwoTowerDataset(Dataset):
    def __init__(self, logs, num_prods, num_negatives=4):
        self.users = logs['user_idx'].values
        self.pos_items = logs['prod_idx'].values
        self.num_prods = num_prods
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        p = self.pos_items[idx]
        
        # Positive 샘플 1개 (실제 클릭/조회) + 정답 1.0
        items = [p]
        labels = [1.0]
        
        # Negative 샘플 4개 (랜덤 추출) + 정답 0.0
        for _ in range(self.num_negatives):
            neg_item = np.random.randint(self.num_prods)
            items.append(neg_item)
            labels.append(0.0)
            
        return torch.tensor([u]*5, dtype=torch.long), torch.tensor(items, dtype=torch.long), torch.tensor(labels, dtype=torch.float32)

dataset = TwoTowerDataset(df_logs, num_prods, num_negatives=4)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 3. Two-Tower 모델 정의
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_prods, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        # User Tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # Item Tower
        self.item_embedding = nn.Embedding(num_prods, embedding_dim)

    def forward(self, user_idx, item_idx):
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        # 내적(Dot Product)으로 두 벡터의 유사도 계산
        dot_product = (u_emb * i_emb).sum(dim=-1)
        # 0~1 사이의 확률값으로 변환
        return torch.sigmoid(dot_product)
    
    def get_item_embeddings(self):
        # 전체 상품의 임베딩만 따로 추출 (서빙용)
        return self.item_embedding.weight.detach().cpu().numpy()

model = TwoTowerModel(num_users, num_prods, embedding_dim=64).to(device)
criterion = nn.BCELoss() # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 모델 학습
epochs = 3
print(f"Two-Tower 모델 학습 시작 (Epochs: {epochs})...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for u_batch, i_batch, label_batch in dataloader:
        u_batch, i_batch, label_batch = u_batch.to(device), i_batch.to(device), label_batch.to(device)
        
        # (Batch, 5) 형태를 1차원으로 펴서 모델에 입력
        u_flat = u_batch.view(-1)
        i_flat = i_batch.view(-1)
        label_flat = label_batch.view(-1)
        
        optimizer.zero_grad()
        outputs = model(u_flat, i_flat)
        loss = criterion(outputs, label_flat)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

# 5. 서빙(Online)을 위한 Item 임베딩 FAISS 인덱스 추출 및 저장
print("상품 임베딩을 FAISS 인덱스로 추출하여 저장합니다...")
os.makedirs('data/models', exist_ok=True)
torch.save(model.state_dict(), 'data/models/two_tower.pth')

item_embeddings = model.get_item_embeddings()
# L2 정규화 (내적 검색을 코사인 유사도처럼 작동하게 만듦)
faiss.normalize_L2(item_embeddings)

# L2 거리 기반 플랫 인덱스 생성 (상품 5만 개 수준이므로 플랫도 충분히 빠름)
item_index = faiss.IndexFlatL2(64)
item_index.add(item_embeddings)

faiss.write_index(item_index, "data/indices/candidate_item.index")
print("✅ Two-Tower 모델 학습 및 후보 생성용 FAISS 인덱스 저장 완료!")