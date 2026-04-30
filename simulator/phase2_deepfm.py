import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 데이터 로드 및 Feature Engineering
print("데이터 로딩 및 피처 병합 중...")
df_users = pd.read_csv('data/users.csv')
df_prods = pd.read_csv('data/products.csv')
df_logs = pd.read_csv('data/train_logs.csv')

# 빠른 테스트를 위해 데이터 사이즈를 조절 (필요시 주석 해제)
# df_logs = df_logs.head(50000)

# 유저와 상품 메타데이터를 로그에 조인 (Context 및 Cross Feature 활용을 위해)
df_merged = df_logs.merge(df_users, on='user_id')
df_merged = df_merged.merge(df_prods, on='product_id')

# 범주형(Categorical) 변수 인코딩
sparse_features = ['user_id', 'product_id', 'persona', 'category_L1', 'price_tier']
encoders = {}
for feat in sparse_features:
    le = LabelEncoder()
    df_merged[feat] = le.fit_transform(df_merged[feat].astype(str))
    encoders[feat] = le

# 정답(Label) 생성: 장바구니(cart)나 구매(purchase)는 1, 단순 조회/검색은 0으로 설정하여 CVR 예측
df_merged['label'] = df_merged['event_type'].apply(lambda x: 1.0 if x in ['cart', 'purchase'] else 0.0)

# 피처 차원(vocab size) 계산
feature_dims = {feat: len(encoders[feat].classes_) for feat in sparse_features}

# 2. Dataset 구성
class DeepFMDataset(Dataset):
    def __init__(self, df, sparse_cols):
        self.X = df[sparse_cols].values
        self.y = df['label'].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

dataset = DeepFMDataset(df_merged, sparse_features)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

# 3. DeepFM 모델 정의
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=16):
        super(DeepFM, self).__init__()
        self.sparse_features = list(feature_dims.keys())
        
        # 임베딩 레이어 (모든 범주형 변수를 동일한 차원수로 임베딩)
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(dim, embedding_dim) for feat, dim in feature_dims.items()
        })
        
        # 1st Order (Linear) 파트
        self.linear = nn.ModuleDict({
            feat: nn.Embedding(dim, 1) for feat, dim in feature_dims.items()
        })
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Deep 파트 (MLP)
        mlp_input_dim = len(feature_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_features)
        
        # [1] 임베딩 추출
        emb_list = [self.embeddings[feat](x[:, i]) for i, feat in enumerate(self.sparse_features)]
        # emb_list: 리스트 내 텐서들 shape (batch, embedding_dim)
        
        # [2] FM (Factorization Machine) 파트
        # 1차 선형 결합 (1st order)
        linear_term = self.bias + sum([self.linear[feat](x[:, i]) for i, feat in enumerate(self.sparse_features)])
        
        # 2차 상호작용 (2nd order) - 교차항 내적의 효율적 계산식
        emb_stack = torch.stack(emb_list, dim=1) # (batch, num_features, embedding_dim)
        sum_of_square = torch.sum(emb_stack, dim=1) ** 2
        square_of_sum = torch.sum(emb_stack ** 2, dim=1)
        fm_term = 0.5 * torch.sum(sum_of_square - square_of_sum, dim=1, keepdim=True)
        
        # [3] Deep 파트
        deep_input = torch.cat(emb_list, dim=1) # (batch, num_features * embedding_dim)
        deep_term = self.mlp(deep_input)
        
        # [4] 최종 출력 결합
        out = linear_term + fm_term + deep_term
        return torch.sigmoid(out).squeeze()

model = DeepFM(feature_dims).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 모델 학습
epochs = 3
print(f"DeepFM 랭킹 모델 학습 시작 (Epochs: {epochs})...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

# 5. 서빙을 위한 모델 저장
os.makedirs('data/models', exist_ok=True)
torch.save(model.state_dict(), 'data/models/deepfm.pth')
print("✅ DeepFM 랭킹 모델 학습 및 가중치 저장 완료! (data/models/deepfm.pth)")