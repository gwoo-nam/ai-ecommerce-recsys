import pandas as pd
import numpy as np
import time
import requests
import random
import os
import yaml # 👈 yaml 패키지 추가 (pip install pyyaml 필요)

print("🚀 [Simulator] 고객 행동 시뮬레이터 가동 준비 중...")

# ---------------------------------------------------------
# [수정] 1. 하드코딩 제거 및 config.yaml 로드
# ---------------------------------------------------------
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        PERSONA_RATIOS = config_data['persona_ratios']
        PERSONA_CONFIG = config_data['persona_configs']
    print("✅ config.yaml 설정 로드 완료")
except FileNotFoundError:
    print("⚠️ config.yaml 파일을 찾을 수 없습니다. 기본값으로 실행합니다.")
    # 파일이 없을 때를 대비한 기본값
    PERSONA_RATIOS = {"트렌드세터": 1.0}
    PERSONA_CONFIG = {"트렌드세터": {"view_to_cart": 0.3, "cart_to_purchase": 0.5, "target_category": "Trending"}}


# 2. 기초 데이터 로드
try:
    df_users = pd.read_csv('data/users.csv')
    df_prods = pd.read_csv('data/products.csv')
except FileNotFoundError:
    print("⚠️ users.csv 또는 products.csv 파일을 찾을 수 없습니다.")
    exit()

LOG_FILE = "data/new_train_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("user_id,product_id,event_type,timestamp\n")

API_URL = "http://127.0.0.1:8000/api/click"
log_count = 0

print("🔥 [Simulator] 실시간 트래픽 생성을 시작합니다! (Ctrl+C로 중단)")

# 3. 무한 루프로 실시간 트래픽 생성
try:
    while True:
        # ---------------------------------------------------------
        # [수정] A. config.yaml의 비율(ratios)을 기반으로 페르소나 선택
        # ---------------------------------------------------------
        personas = list(PERSONA_RATIOS.keys())
        probabilities = list(PERSONA_RATIOS.values())
        chosen_persona = np.random.choice(personas, p=probabilities)
        
        # 선택된 페르소나를 가진 유저 중에서 한 명 뽑기
        user = df_users[df_users['persona'] == chosen_persona].sample(1).iloc[0]
        user_id = user['user_id']
        persona = user['persona']
        
        # ---------------------------------------------------------
        # [수정] B. config.yaml에 정의된 타겟 카테고리로 상품 검색
        # ---------------------------------------------------------
        config = PERSONA_CONFIG.get(persona, PERSONA_CONFIG["신중탐색"])
        target_keyword = config["target_category"]
        
        # 만약 타겟 카테고리가 All이 아니면 해당 키워드가 포함된 상품 검색
        if target_keyword != "All":
            target_products = df_prods[df_prods['product_name'].str.contains(target_keyword, case=False, na=False)]
            # 해당 키워드 상품이 없으면 전체에서 랜덤 추출
            if target_products.empty:
                target_products = df_prods
        else:
            target_products = df_prods
            
        product = target_products.sample(1).iloc[0]
        product_id = product['product_id']

        # C. 이벤트 타입 결정 (페르소나별 전환율 적용)
        event_type = "view"
        if random.random() < config["view_to_cart"]:
            event_type = "cart"
            if random.random() < config["cart_to_purchase"]:
                event_type = "purchase"

        # Redis API 호출
        try:
            requests.post(API_URL, data={"user_id": user_id, "item_id": product_id}, timeout=1)
        except requests.exceptions.RequestException:
            pass

        # 로그 기록
        timestamp = int(time.time())
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("user_id,product_id,event_type,timestamp\n")
            f.write(f"{user_id},{product_id},{event_type},{timestamp}\n")
        
        log_count += 1
        
        if log_count % 100 == 0:
            print(f"📊 [Simulator] 생성된 누적 로그: {log_count}건 (최근: {user_id}가 {product_id}를 {event_type}함)")

        time.sleep(0.05) 

except KeyboardInterrupt:
    print(f"\n🛑 [Simulator] 시뮬레이터 중지됨. 총 {log_count}건의 로그를 생성했습니다.")