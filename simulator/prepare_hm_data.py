import os
import yaml
import pandas as pd
import numpy as np

print("🚀 H&M 데이터 정밀 전처리 및 페르소나 기반 퍼널 증강 파이프라인 시작...")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = "data"
RAW_DIR = "data/raw"

ARTICLES_PATH = f"{RAW_DIR}/articles.csv"
CUSTOMERS_PATH = f"{RAW_DIR}/customers.csv"
TRANSACTIONS_PATH = f"{RAW_DIR}/transactions_train.csv"
CONFIG_PATH = "config.yaml"

os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_PERSONA_RATIOS = {
    "가성비추구": 0.25,
    "브랜드충성": 0.15,
    "신중탐색": 0.10,
    "실용주의자": 0.25,
    "충동구매": 0.10,
    "트렌드세터": 0.15,
}

DEFAULT_PERSONA_CONFIGS = {
    "가성비추구": {
        "match_field": "any",
        "match_values": [],
        "price_sensitivity": "high",
        "search_prob": 0.7,
        "view_to_cart": 0.10,
        "cart_to_purchase": 0.50,
    },
    "브랜드충성": {
        "match_field": "category_L1",
        "match_values": ["Ladieswear", "Menswear"],
        "price_sensitivity": "low",
        "search_prob": 0.3,
        "view_to_cart": 0.12,
        "cart_to_purchase": 0.45,
    },
    "신중탐색": {
        "match_field": "any",
        "match_values": [],
        "price_sensitivity": "high",
        "search_prob": 0.8,
        "view_to_cart": 0.05,
        "cart_to_purchase": 0.20,
    },
    "실용주의자": {
        "match_field": "category_L3",
        "match_values": ["Garment Upper body", "Underwear", "Socks & Tights"],
        "price_sensitivity": "medium",
        "search_prob": 0.5,
        "view_to_cart": 0.08,
        "cart_to_purchase": 0.50,
    },
    "충동구매": {
        "match_field": "category_L3",
        "match_values": ["Accessories", "Bags", "Shoes"],
        "price_sensitivity": "medium",
        "search_prob": 0.2,
        "view_to_cart": 0.20,
        "cart_to_purchase": 0.30,
    },
    "트렌드세터": {
        "match_field": "category_L3",
        "match_values": ["Trousers", "Garment Upper body"],
        "price_sensitivity": "medium",
        "search_prob": 0.6,
        "view_to_cart": 0.15,
        "cart_to_purchase": 0.40,
    },
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        persona_ratios = cfg.get("persona_ratios", DEFAULT_PERSONA_RATIOS)
        persona_configs = cfg.get("persona_configs", DEFAULT_PERSONA_CONFIGS)

        print("✅ config.yaml 로드 완료")
        return persona_ratios, persona_configs

    print("⚠️ config.yaml 없음 → 기본 페르소나 설정 사용")
    return DEFAULT_PERSONA_RATIOS, DEFAULT_PERSONA_CONFIGS


def normalize_ratio_dict(ratio_dict):
    personas = list(ratio_dict.keys())
    probs = np.array([float(ratio_dict[p]) for p in personas], dtype=float)
    probs = probs / probs.sum()
    return personas, probs


def apply_price_filter(pool, sensitivity):
    if "price_tier" not in pool.columns:
        return pool

    if sensitivity == "high":
        filtered = pool[pool["price_tier"] == "low"]
    elif sensitivity == "low":
        filtered = pool[pool["price_tier"].isin(["medium", "high"])]
    else:
        filtered = pool

    return filtered if len(filtered) > 0 else pool


def build_candidate_pool(products_df, persona_cfg):
    field = persona_cfg.get("match_field", "any")
    values = persona_cfg.get("match_values", []) or []

    if field == "any" or not values:
        pool = products_df.copy()
    elif field in products_df.columns:
        pattern = "|".join([str(v) for v in values])
        pool = products_df[
            products_df[field].astype(str).str.contains(pattern, case=False, na=False)
        ].copy()
    else:
        pool = products_df.copy()

    if pool.empty:
        pool = products_df.copy()

    pool = apply_price_filter(pool, persona_cfg.get("price_sensitivity", "medium"))

    if pool.empty:
        pool = products_df.copy()

    return pool


def sample_products_from_pool(pool, size, replace=True):
    return pool["product_id"].sample(
        n=size,
        replace=replace,
        random_state=np.random.randint(0, 1_000_000_000)
    ).values


print("1. 설정 로딩 중...")
PERSONA_RATIOS, PERSONA_CONFIGS = load_config()
PERSONAS, PERSONA_PROBS = normalize_ratio_dict(PERSONA_RATIOS)

print("2. 원본 데이터 로딩 중...")
articles = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
customers = pd.read_csv(CUSTOMERS_PATH)
trans = pd.read_csv(TRANSACTIONS_PATH, dtype={"article_id": str})

print("3. 실제 거래 기록에서 상품별 평균 가격 계산 중...")
trans["real_price"] = trans["price"] * 1_000_000
item_avg_price = trans.groupby("article_id")["real_price"].mean().reset_index()

print("4. Active 유저 및 핵심 상품 샘플링 중...")
user_counts = trans["customer_id"].value_counts()
active_users = user_counts[user_counts >= 5].index
trans_active = trans[trans["customer_id"].isin(active_users)].copy()

item_counts = trans_active["article_id"].value_counts()
top_items = item_counts.head(50000).index

final_trans = trans_active[trans_active["article_id"].isin(top_items)].copy()
final_articles = articles[articles["article_id"].isin(top_items)].copy()

print("5. 가격 등급 price_tier 생성 중...")
final_articles = final_articles.merge(item_avg_price, on="article_id", how="left")
final_articles["real_price"] = final_articles["real_price"].fillna(final_articles["real_price"].median())

final_articles["price_tier"] = pd.qcut(
    final_articles["real_price"],
    q=3,
    labels=["low", "medium", "high"],
    duplicates="drop"
)

final_articles["price_tier"] = final_articles["price_tier"].astype(str)

print("6. products.csv 생성 중...")
products = pd.DataFrame({
    "product_id": "P" + final_articles["article_id"],
    "category_L1": final_articles["index_group_name"].fillna("unknown"),
    "category_L2": final_articles["index_name"].fillna("unknown"),
    "category_L3": final_articles["product_group_name"].fillna("unknown"),
    "product_name": (
        final_articles["prod_name"].fillna("")
        + " - "
        + final_articles["colour_group_name"].fillna("")
        + " ("
        + final_articles["detail_desc"].fillna("")
        + ")"
    ),
    "price": final_articles["real_price"].astype(int),
    "price_tier": final_articles["price_tier"].fillna("medium"),
})

products.to_csv(f"{DATA_DIR}/products.csv", index=False)
print(f"  ✅ products.csv 저장 완료: {len(products):,}개")

print("7. users.csv 생성 중...")
sampled_users = pd.Series(final_trans["customer_id"].unique()).head(10000)
customers_filtered = customers[customers["customer_id"].isin(sampled_users)].copy()

customers_filtered["persona"] = np.random.choice(
    PERSONAS,
    size=len(customers_filtered),
    p=PERSONA_PROBS
)

users = pd.DataFrame({
    "user_id": "U" + customers_filtered["customer_id"].str[:10],
    "persona": customers_filtered["persona"],
})

users.to_csv(f"{DATA_DIR}/users.csv", index=False)
print(f"  ✅ users.csv 저장 완료: {len(users):,}명")

print("8. 페르소나별 상품 후보 풀 생성 중...")
persona_pools = {}
for persona in PERSONAS:
    pcfg = PERSONA_CONFIGS.get(persona, {})
    pool = build_candidate_pool(products, pcfg)
    persona_pools[persona] = pool
    print(f"  {persona:<8} → {len(pool):,}개 상품")

user_persona_map = users.set_index("user_id")["persona"].to_dict()

print("9. 구매 로그 기반 성공 퍼널 생성 중...")
final_trans_sampled = final_trans[final_trans["customer_id"].isin(sampled_users)].copy()

purchase_logs = pd.DataFrame({
    "user_id": "U" + final_trans_sampled["customer_id"].str[:10],
    "product_id": "P" + final_trans_sampled["article_id"],
    "event_type": "purchase",
    "timestamp": pd.to_datetime(final_trans_sampled["t_dat"]),
})

purchase_logs = purchase_logs[
    purchase_logs["user_id"].isin(users["user_id"])
    & purchase_logs["product_id"].isin(products["product_id"])
].copy()

purchase_logs["is_bounced"] = 0
purchase_logs["is_abandoned"] = 0
purchase_logs["funnel_type"] = "success_purchase"
purchase_logs["event_weight"] = 3.0

cart_for_purchase = purchase_logs.copy()
cart_for_purchase["event_type"] = "cart"
cart_for_purchase["funnel_type"] = "success_cart"
cart_for_purchase["event_weight"] = 2.0
cart_for_purchase["timestamp"] = cart_for_purchase["timestamp"] - pd.to_timedelta(10, unit="m")

view_for_purchase = purchase_logs.copy()
view_for_purchase["event_type"] = "view"
view_for_purchase["funnel_type"] = "success_view"
view_for_purchase["event_weight"] = 1.0
view_for_purchase["timestamp"] = view_for_purchase["timestamp"] - pd.to_timedelta(30, unit="m")

print("10. 페르소나 기반 이탈 view / 장바구니 유기 로그 생성 중...")
base_for_negative = purchase_logs[["user_id", "timestamp"]].copy()

bounced_views = base_for_negative.sample(
    frac=3.0,
    replace=True,
    random_state=RANDOM_SEED
).copy()

abandoned_carts = base_for_negative.sample(
    frac=0.5,
    replace=True,
    random_state=RANDOM_SEED + 1
).copy()


def assign_persona_products(df):
    result = []
    for user_id in df["user_id"].values:
        persona = user_persona_map.get(user_id, "신중탐색")
        pool = persona_pools.get(persona, products)
        product_id = pool["product_id"].sample(
            n=1,
            replace=True,
            random_state=np.random.randint(0, 1_000_000_000)
        ).iloc[0]
        result.append(product_id)
    return result


bounced_views["product_id"] = assign_persona_products(bounced_views)
bounced_views["event_type"] = "view"
bounced_views["is_bounced"] = 1
bounced_views["is_abandoned"] = 0
bounced_views["funnel_type"] = "bounced_view"
bounced_views["event_weight"] = 0.3
bounced_views["timestamp"] = bounced_views["timestamp"] + pd.to_timedelta(
    np.random.randint(1, 120, size=len(bounced_views)),
    unit="m"
)

abandoned_carts["product_id"] = assign_persona_products(abandoned_carts)
abandoned_carts["event_type"] = "cart"
abandoned_carts["is_bounced"] = 0
abandoned_carts["is_abandoned"] = 1
abandoned_carts["funnel_type"] = "abandoned_cart"
abandoned_carts["event_weight"] = 1.2
abandoned_carts["timestamp"] = abandoned_carts["timestamp"] + pd.to_timedelta(
    np.random.randint(1, 180, size=len(abandoned_carts)),
    unit="m"
)

abandoned_cart_views = abandoned_carts.copy()
abandoned_cart_views["event_type"] = "view"
abandoned_cart_views["is_bounced"] = 0
abandoned_cart_views["is_abandoned"] = 1
abandoned_cart_views["funnel_type"] = "abandoned_cart_view"
abandoned_cart_views["event_weight"] = 0.8
abandoned_cart_views["timestamp"] = abandoned_cart_views["timestamp"] - pd.to_timedelta(20, unit="m")

print("11. 페르소나별 search 이벤트 생성 중...")
search_logs_list = []

all_candidate_logs = pd.concat(
    [
        view_for_purchase[["user_id", "product_id", "timestamp"]],
        bounced_views[["user_id", "product_id", "timestamp"]],
        abandoned_cart_views[["user_id", "product_id", "timestamp"]],
    ],
    ignore_index=True
)

for persona in PERSONAS:
    pcfg = PERSONA_CONFIGS.get(persona, {})
    search_prob = float(pcfg.get("search_prob", 0.0))

    persona_users = users[users["persona"] == persona]["user_id"]
    persona_events = all_candidate_logs[all_candidate_logs["user_id"].isin(persona_users)]

    if len(persona_events) == 0 or search_prob <= 0:
        continue

    sampled = persona_events.sample(
        frac=min(search_prob, 1.0),
        replace=False,
        random_state=RANDOM_SEED + len(search_logs_list)
    ).copy()

    sampled["event_type"] = "search"
    sampled["is_bounced"] = 0
    sampled["is_abandoned"] = 0
    sampled["funnel_type"] = "search"
    sampled["event_weight"] = 0.5
    sampled["timestamp"] = sampled["timestamp"] - pd.to_timedelta(5, unit="m")

    search_logs_list.append(sampled)

if search_logs_list:
    search_logs = pd.concat(search_logs_list, ignore_index=True)
else:
    search_logs = pd.DataFrame(columns=[
        "user_id", "product_id", "timestamp", "event_type",
        "is_bounced", "is_abandoned", "funnel_type", "event_weight"
    ])

print("12. 전체 로그 병합 및 시간순 정렬 중...")
logs = pd.concat(
    [
        search_logs,
        view_for_purchase,
        cart_for_purchase,
        purchase_logs,
        bounced_views,
        abandoned_cart_views,
        abandoned_carts,
    ],
    ignore_index=True
)

logs = logs[
    logs["user_id"].isin(users["user_id"])
    & logs["product_id"].isin(products["product_id"])
].copy()

logs["timestamp"] = pd.to_datetime(logs["timestamp"])
logs = logs.sort_values("timestamp").reset_index(drop=True)

logs = logs[
    [
        "user_id",
        "product_id",
        "event_type",
        "timestamp",
        "is_bounced",
        "is_abandoned",
        "funnel_type",
        "event_weight",
    ]
]

print("13. Train / Valid / Test 시간 기반 8:1:1 분할 중...")
train_cutoff = logs["timestamp"].quantile(0.8)
valid_cutoff = logs["timestamp"].quantile(0.9)

train_logs = logs[logs["timestamp"] <= train_cutoff].copy()
valid_logs = logs[
    (logs["timestamp"] > train_cutoff)
    & (logs["timestamp"] <= valid_cutoff)
].copy()
test_logs = logs[logs["timestamp"] > valid_cutoff].copy()

train_logs.to_csv(f"{DATA_DIR}/train_logs.csv", index=False)
valid_logs.to_csv(f"{DATA_DIR}/valid_logs.csv", index=False)
test_logs.to_csv(f"{DATA_DIR}/test_logs.csv", index=False)

print("14. 데이터 품질 요약")
print(f"  전체 로그: {len(logs):,}건")
print(f"  Train   : {len(train_logs):,}건 ({train_logs['timestamp'].min().date()} ~ {train_logs['timestamp'].max().date()})")
print(f"  Valid   : {len(valid_logs):,}건 ({valid_logs['timestamp'].min().date()} ~ {valid_logs['timestamp'].max().date()})")
print(f"  Test    : {len(test_logs):,}건 ({test_logs['timestamp'].min().date()} ~ {test_logs['timestamp'].max().date()})")

print("\n  이벤트 분포")
print(logs["event_type"].value_counts())

print("\n  퍼널 타입 분포")
print(logs["funnel_type"].value_counts())

print("\n  페르소나 분포")
print(users["persona"].value_counts(normalize=True).round(4))

print("\n  가격대 분포")
print(products["price_tier"].value_counts(normalize=True).round(4))

print("\n✅ 데이터 전처리 및 페르소나 기반 로그 생성 완료!")
print("   - data/products.csv")
print("   - data/users.csv")
print("   - data/train_logs.csv")
print("   - data/valid_logs.csv")
print("   - data/test_logs.csv")
