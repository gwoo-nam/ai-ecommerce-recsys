import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [Retrain Job] CT 파이프라인 워커 노드 가동 (Device: {device})")

# ---------------------------------------------------------
# 1. API 서버와 동일한 DeepFM 모델 아키텍처 선언
# ---------------------------------------------------------
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
        deep_term = self.mlp(torch.cat(emb_list, dim=1))
        return torch.sigmoid(linear_term + fm_term + deep_term).squeeze()

# ---------------------------------------------------------
# 2. 데이터 및 인코더 로드 (차원 일치를 위해 필수)
# ---------------------------------------------------------
print("1️⃣ 데이터 및 기존 인코더 복원 중...")
df_users = pd.read_csv('data/users.csv')
df_prods = pd.read_csv('data/products.csv')
df_logs_old = pd.read_csv('data/train_logs.csv') # 기존 차원 계산용
df_new_logs = pd.read_csv('data/new_train_logs.csv') # 새로 쌓인 1만건의 트렌드!

# 기존 모델과 똑같은 차원을 맞추기 위해 인코더 재생성
sparse_features = ['user_id', 'product_id', 'persona', 'category_L1', 'price_tier']
encoders = {}
temp_merged = df_logs_old.merge(df_users, on='user_id').merge(df_prods, on='product_id')
for feat in sparse_features:
    le = LabelEncoder()
    temp_merged[feat] = le.fit_transform(temp_merged[feat].astype(str))
    encoders[feat] = le

feature_dims = {feat: len(encoders[feat].classes_) for feat in sparse_features}

# ---------------------------------------------------------
# 3. 신규 로그 데이터 전처리
# ---------------------------------------------------------
print("2️⃣ 신규 트렌드 데이터 전처리 중...")
df_merged_new = df_new_logs.merge(df_users, on='user_id').merge(df_prods, on='product_id')
df_merged_new['label'] = df_merged_new['event_type'].apply(lambda x: 1.0 if x in ['cart', 'purchase'] else 0.0)

X_pred = np.zeros((len(df_merged_new), len(sparse_features)), dtype=int)
for i, feat in enumerate(sparse_features):
    classes = encoders[feat].classes_
    vals = df_merged_new[feat].astype(str).values
    # 학습 때 보지 못한 완전히 새로운 아이템/유저가 오면 0(기본값)으로 처리하여 에러 방지
    X_pred[:, i] = np.where(np.isin(vals, classes), encoders[feat].transform(vals), 0)

class NewLogDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

dataloader = DataLoader(NewLogDataset(X_pred, df_merged_new['label'].values), batch_size=512, shuffle=True)

# ---------------------------------------------------------
# 4. 기존 모델 가중치 로드 및 파인튜닝 (Fine-tuning)
# ---------------------------------------------------------
print("3️⃣ 기존 DeepFM 모델 로드 및 미세조정(Fine-tuning) 시작...")
model = DeepFM(feature_dims).to(device)

# 과거의 똑똑한 뇌(가중치)를 그대로 이식
model_path = 'data/models/deepfm.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("⚠️ 기존 모델이 없어 처음부터 학습합니다.")

criterion = nn.BCELoss()

# 💡 MLOps 핵심: 기존 지식을 잊어버리지 않도록 학습률(Learning Rate)을 0.0001로 매우 낮게 설정!
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# 딱 1 Epoch만 빠르고 가볍게 돌려서 최신 트렌드만 살짝 반영
model.train()
for epoch in range(10): # 10번 반복 학습
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} 완료 (Loss: {total_loss/len(dataloader):.4f})")

# ---------------------------------------------------------
# 5. 모델 덮어쓰기 (API 서버가 즉시 사용할 수 있도록)
# ---------------------------------------------------------
torch.save(model.state_dict(), model_path)
print("💾 새로운 가중치가 data/models/deepfm.pth 에 성공적으로 덮어씌워졌습니다!")

def calculate_offline_metrics(model, test_loader, device, k=50):
    """실제 테스트 데이터를 돌려 HitRate와 NDCG를 계산합니다."""
    model.eval()
    hits = 0
    ndcgs = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            labels = y_batch.numpy()
            
            # 예측값 상위 K개 안에 실제 정답(Label=1)이 있는지 확인
            for i in range(len(labels)):
                if labels[i] == 1: # 정답인 경우만 평가
                    total += 1
                    # 간단한 구현을 위해 배치 내 랭킹 시뮬레이션
                    rank = np.where(np.argsort(preds)[::-1] == i)[0][0] + 1
                    if rank <= k:
                        hits += 1
                        ndcgs += 1 / np.log2(rank + 1)
    
    return {
        "hitrate": hits / total if total > 0 else 0,
        "ndcg": ndcgs / total if total > 0 else 0
    }

# ---------------------------------------------------------
# [핵심] 모든 지표를 계산하여 metrics.json으로 저장
# ---------------------------------------------------------
def update_metrics_json(mrr_val, ndcg_10_val, recall_val, auc_val, hitrate_val, ndcg_50_val, coverage_val, latency):
    final_metrics = {
        "search": {
            "mrr": mrr_val,       # CLIP+FAISS 실제 계산값
            "ndcg_10": ndcg_10_val,
            "latency_ms": latency["search"]
        },
        "recommend": {
            "recall_300": recall_val, # Two-Tower 실제 계산값
            "auc": auc_val,           # DeepFM 실제 계산값
            "hitrate_50": hitrate_val, # Re-ranking 결과
            "ndcg_50": ndcg_50_val,
            "coverage": coverage_val,
            "latency_ms": latency["recommend"]
        }
    }
    
    with open('data/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=4)