# 1. 파이썬 3.10 기반 이미지 사용
FROM python:3.10-slim

# 2. 작업 폴더 지정
WORKDIR /app

# 3. 필수 시스템 패키지 설치 (PyTorch, FAISS 등 설치 시 필요)
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 나머지 모든 소스코드 복사
COPY . .