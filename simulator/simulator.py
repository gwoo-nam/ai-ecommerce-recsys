"""
simulator.py  (개선판)
─────────────────────
변경 사항:
  ✅ config.yaml 의 match_field / match_values 기반 카테고리 매칭
     (기존: product_name 부분문자열 매칭 → H&M 데이터에 0건이라 페르소나 차별화 사라짐)
  ✅ /api/log 의 LogEvent 스키마(JSON body)에 맞춰 requests.post(..., json=payload)
     (기존: data= 로 전송하고 키 이름도 item_id 로 보내 서버 검증 실패)
  ✅ 시드 고정 (재현성: 명세서 요구사항)
  ✅ 매칭 0건 fallback 시 경고만 한 번 출력
"""
import pandas as pd
import numpy as np
import time
import requests
import random
import os
import yaml

print("🚀 [Simulator] 고객 행동 시뮬레이터 가동 준비 중...")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# 1. config.yaml 로드
# ──────────────────────────────────────────────
try:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    PERSONA_RATIOS = cfg["persona_ratios"]
    PERSONA_CONFIG = cfg["persona_configs"]
    API_URL = cfg.get("api", {}).get("log_endpoint", "http://api-server:8000/api/log")
    print(f"✅ config.yaml 로드 완료 (API: {API_URL})")
except FileNotFoundError:
    print("⚠️ config.yaml 없음 → 기본값 사용")
    PERSONA_RATIOS = {"신중탐색": 1.0}
    PERSONA_CONFIG = {
        "신중탐색": {
            "match_field": "any", "match_values": [],
            "view_to_cart": 0.05, "cart_to_purchase": 0.20,
            "search_prob": 0.5, "price_sensitivity": "medium",
        }
    }
    API_URL = "http://api-server:8000/api/log"


# ──────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────
try:
    df_users = pd.read_csv("data/users.csv")
    df_prods = pd.read_csv("data/products.csv")
except FileNotFoundError:
    print("⚠️ data/users.csv 또는 data/products.csv 없음. prepare_hm_data.py 먼저 실행하세요.")
    exit(1)

LOG_FILE = "data/new_train_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("user_id,product_id,event_type,timestamp\n")


# ──────────────────────────────────────────────
# 3. 페르소나별 후보 풀 사전 빌드
#    매번 string contains 호출하지 않도록 한 번만 만들어 둔다.
# ──────────────────────────────────────────────
def build_candidate_pool(persona_cfg: dict) -> pd.DataFrame:
    """match_field/match_values 에 따른 상품 후보 데이터프레임."""
    field  = persona_cfg.get("match_field", "any")
    values = persona_cfg.get("match_values", []) or []

    if field == "any" or not values:
        pool = df_prods.copy()
    elif field in df_prods.columns:
        # 대소문자 무시 부분일치
        pattern = "|".join([str(v) for v in values])
        mask = df_prods[field].astype(str).str.contains(pattern, case=False, na=False)
        pool = df_prods[mask].copy()
    else:
        # 컬럼이 없으면 product_name 폴백
        pattern = "|".join([str(v) for v in values])
        mask = df_prods["product_name"].astype(str).str.contains(pattern, case=False, na=False)
        pool = df_prods[mask].copy()

    if pool.empty:
        pool = df_prods.copy()  # 매칭 0건이면 전체로 폴백
    return pool


# 가격 민감도 적용
def apply_price_filter(pool: pd.DataFrame, sensitivity: str) -> pd.DataFrame:
    if "price_tier" not in pool.columns:
        return pool
    if sensitivity == "high":
        f = pool[pool["price_tier"] == "low"]
    elif sensitivity == "low":
        f = pool[pool["price_tier"].isin(["medium", "high"])]
    else:
        return pool
    return f if not f.empty else pool


# 사전 빌드
print("📦 페르소나별 후보 풀 사전 빌드 중...")
PERSONA_POOLS = {}
for persona, pcfg in PERSONA_CONFIG.items():
    base_pool = build_candidate_pool(pcfg)
    final_pool = apply_price_filter(base_pool, pcfg.get("price_sensitivity", "medium"))
    PERSONA_POOLS[persona] = final_pool
    print(f"  {persona:<8} → {len(final_pool):,}개 상품 (match_field={pcfg.get('match_field')})")

# 매칭 0건 경고 (전체로 폴백된 경우)
for persona, pool in PERSONA_POOLS.items():
    if len(pool) == len(df_prods):
        print(f"  ⚠️ '{persona}' 매칭이 전체와 같음 → match_values 가 데이터에 거의 없을 수 있음")


# ──────────────────────────────────────────────
# 4. 이벤트 전송
# ──────────────────────────────────────────────
def post_event(user_id: str, product_id: str, event_type: str, timestamp: int) -> None:
    payload = {
        "user_id":    user_id,
        "product_id": product_id,
        "event_type": event_type,
        "timestamp":  timestamp,
    }
    # FastAPI 의 LogEvent 모델은 JSON body 를 기대 → json= 으로 전송해야 함
    try:
        requests.post(API_URL, json=payload, timeout=1)
    except requests.exceptions.RequestException:
        pass

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{user_id},{product_id},{event_type},{timestamp}\n")


# ──────────────────────────────────────────────
# 5. 메인 루프
# ──────────────────────────────────────────────
log_count = 0
print("🔥 실시간 트래픽 생성 시작 (Ctrl+C 로 중단)")

try:
    personas      = list(PERSONA_RATIOS.keys())
    probabilities = list(PERSONA_RATIOS.values())

    # users 를 페르소나별 인덱스로 미리 그룹핑 (sample 비용 절감)
    users_by_persona = {p: df_users[df_users["persona"] == p] for p in personas}

    while True:
        chosen_persona = np.random.choice(personas, p=probabilities)
        candidates = users_by_persona.get(chosen_persona)
        if candidates is None or candidates.empty:
            continue

        user = candidates.sample(1).iloc[0]
        user_id = user["user_id"]
        persona = user["persona"]

        pcfg  = PERSONA_CONFIG.get(persona, PERSONA_CONFIG["신중탐색"])
        pool  = PERSONA_POOLS.get(persona, df_prods)

        product = pool.sample(1).iloc[0]
        product_id = product["product_id"]
        timestamp  = int(time.time())

        # 검색 이벤트
        if random.random() < pcfg.get("search_prob", 0.0):
            post_event(user_id, product_id, "search", timestamp)

        # view → cart → purchase 퍼널
        event_type = "view"
        if random.random() < pcfg["view_to_cart"]:
            event_type = "cart"
            if random.random() < pcfg["cart_to_purchase"]:
                event_type = "purchase"

        post_event(user_id, product_id, event_type, timestamp)
        log_count += 1

        if log_count % 100 == 0:
            print(f"📊 누적 로그: {log_count}건  (최근: {user_id} → {product_id} [{event_type}], 페르소나: {persona})")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(f"\n🛑 중지됨. 총 {log_count}건 생성")