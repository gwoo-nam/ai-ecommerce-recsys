"""
phase4_dashboard.py
────────────────────
AI 추천 시스템 통합 모니터링 대시보드 (Streamlit)

탭 구성
  Tab 1 │ 실시간 비즈니스 지표    ─ new_train_logs.csv / archive/*.csv 기반
  Tab 2 │ 모델 오프라인 평가      ─ data/metrics.json 기반 (실제 계산값)
  Tab 3 │ A/B 테스트 엔진        ─ Chi-square + 95% 신뢰구간 실시간 계산

실행 방법:
  streamlit run phase4_dashboard.py
"""

import json
import os
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency, norm

# ──────────────────────────────────────────────────────────────────────────────
# 0. 페이지 기본 설정
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI E-Commerce 대시보드",
    page_icon="🛒",
    layout="wide",
)
st.title("🛒 AI 추천 시스템 통합 모니터링 대시보드")
st.markdown("실제 유저 로그를 기반으로 추천 시스템의 비즈니스 성과와 통계적 유의성을 분석합니다.")


# ──────────────────────────────────────────────────────────────────────────────
# 1. 공통 데이터 로드  (TTL=60초 캐시)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_base_data():
    users = pd.read_csv("data/users.csv")
    prods = pd.read_csv("data/products.csv")
    return users, prods


@st.cache_data(ttl=60)
def load_live_logs():
    """실시간으로 쌓이는 new_train_logs.csv 를 읽습니다."""
    path = "data/new_train_logs.csv"
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_latest_archive_logs():
    """가장 최근 CT 재학습에 사용된 archive CSV 를 읽습니다."""
    files = glob.glob("data/archive/*.csv")
    if not files:
        return pd.DataFrame()
    latest = max(files, key=os.path.getctime)
    try:
        return pd.read_csv(latest), os.path.basename(latest)
    except Exception:
        return pd.DataFrame(), ""


@st.cache_data(ttl=30)
def load_metrics():
    """metrics.json 을 읽어 반환합니다. 없으면 None."""
    path = "data/metrics.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


df_users, df_prods = load_base_data()
df_logs = load_live_logs()

# ──────────────────────────────────────────────────────────────────────────────
# 탭 레이아웃
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    [
        "📈 실시간 비즈니스 지표 (Live)",
        "🎯 모델 오프라인 평가 (Metrics)",
        "🧪 A/B 테스트 엔진 (Experiment)",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 ─ 실시간 비즈니스 지표
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 현재 쇼핑몰 트래픽 현황")

    if df_logs.empty:
        st.warning(
            "실시간 로그(data/new_train_logs.csv)를 찾을 수 없습니다. "
            "simulator.py 를 먼저 실행하세요."
        )
    else:
        # ── 핵심 지표 카드 ─────────────────────────────────────────────────
        event_counts = df_logs["event_type"].value_counts()
        searches  = int(event_counts.get("search",   0))  # 👈 추가!
        views     = int(event_counts.get("view",     0))
        carts     = int(event_counts.get("cart",     0))
        purchases = int(event_counts.get("purchase", 0))
        # CVR = 구매 / 전체 이벤트 (0~100% 보장)
        total_evt = views + carts + purchases
        cur_cvr   = (purchases / total_evt * 100) if total_evt > 0 else 0.0

        # 아카이브 최근 1만 건 CVR
        archive_result = load_latest_archive_logs()
        if isinstance(archive_result, tuple):
            df_archive, arch_name = archive_result
        else:
            df_archive, arch_name = archive_result, ""

        if not df_archive.empty:
            arch_purchases = int((df_archive["event_type"] == "purchase").sum())
            arch_total     = len(df_archive)
            latest_cvr     = (arch_purchases / arch_total * 100) if arch_total > 0 else 0.0
        else:
            latest_cvr = 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("총 가입 유저",        f"{len(df_users):,}명")
        c2.metric("등록 상품 수",        f"{len(df_prods):,}개")
        c3.metric("누적 실시간 로그",    f"{len(df_logs):,}건")
        c4.metric("현재 실시간 CVR",     f"{cur_cvr:.2f}%")
        c5.metric(
            "🔥 최근 CT 완료 1만 건 CVR",
            f"{latest_cvr:.2f}%",
            help=f"출처: {arch_name}",
        )

        st.divider()

        # ── 구매 퍼널 차트 ──────────────────────────────────────────────────
        col_funnel, col_persona = st.columns([1, 1])

        with col_funnel:
            st.markdown("#### 사용자 행동 퍼널")
            fig_funnel = go.Figure(
                go.Funnel(
                    y=["상품 검색 (Search)", "상품 조회 (View)", "장바구니 (Cart)", "최종 구매 (Purchase)"], # 👈 추가
                    x=[searches, views, carts, purchases], # 👈 추가
                    textinfo="value+percent initial",
                    marker={"color": ["#FFA726", "#5C6BC0", "#42A5F5", "#66BB6A"]}, # 컬러맵 4개로 맞춤
                )
            )
            fig_funnel.update_layout(
                title="행동 퍼널 분석 (실시간 로그 기반)",
                height=360,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_funnel, width='stretch')

        # ── 페르소나별 이벤트 분포 ───────────────────────────────────────────
        with col_persona:
            st.markdown("#### 페르소나별 이벤트 분포")
            if "user_id" in df_logs.columns:
                df_persona_log = df_logs.merge(
                    df_users[["user_id", "persona"]], on="user_id", how="left"
                )
                persona_counts = (
                    df_persona_log.groupby(["persona", "event_type"])
                    .size()
                    .reset_index(name="count")
                )
                if not persona_counts.empty:
                    fig_persona = px.bar(
                        persona_counts,
                        x="persona",
                        y="count",
                        color="event_type",
                        barmode="stack",
                        color_discrete_map={
                            "view": "#5C6BC0",
                            "cart": "#42A5F5",
                            "purchase": "#66BB6A",
                        },
                        labels={"count": "이벤트 수", "persona": "페르소나"},
                        title="페르소나별 행동 분포",
                        height=360,
                    )
                    fig_persona.update_layout(margin=dict(t=40, b=20))
                    st.plotly_chart(fig_persona, width='stretch')
                else:
                    st.info("페르소나 데이터를 집계 중입니다...")
            else:
                st.info("user_id 컬럼이 없습니다.")

        st.divider()

        # ── 카테고리별 트렌드 (실시간 purchase 기준) ─────────────────────────
        st.markdown("#### 🔥 실시간 카테고리 트렌드 (구매 기준)")
        if "product_id" in df_logs.columns:
            df_purchase_live = df_logs[df_logs["event_type"] == "purchase"].copy()
            if not df_purchase_live.empty:
                df_trend = df_purchase_live.merge(
                    df_prods[["product_id", "category_L1"]], on="product_id", how="left"
                )
                cat_counts = (
                    df_trend["category_L1"]
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                cat_counts.columns = ["category_L1", "purchase_count"]

                fig_cat = px.bar(
                    cat_counts,
                    x="purchase_count",
                    y="category_L1",
                    orientation="h",
                    color="purchase_count",
                    color_continuous_scale="Blues",
                    labels={"purchase_count": "구매 수", "category_L1": "대분류"},
                    title="카테고리 대분류별 실시간 구매 수 Top-10",
                )
                fig_cat.update_layout(height=380, margin=dict(t=40, b=20))
                st.plotly_chart(fig_cat, width='stretch')
            else:
                st.info("아직 purchase 이벤트가 없습니다.")

        # ── 새로고침 버튼 ──────────────────────────────────────────────────
        if st.button("🔄 데이터 새로고침", width='content'):
            st.cache_data.clear()
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 ─ 모델 오프라인 평가
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎯 모델 오프라인 평가 리포트 (실제 계산값 기반)")

    m = load_metrics()

    if m is None:
        st.error(
            "**data/metrics.json 파일이 없습니다.**\n\n"
            "아래 순서로 실행하면 지표가 자동 생성됩니다.\n"
            "1. `python phase4_offline_eval.py`  — CLIP+FAISS / Two-Tower 지표 계산\n"
            "2. `python phase4_retrain_job.py`   — DeepFM 지표 계산 + metrics.json 저장\n\n"
            "또는 CT 파이프라인이 1만 건을 감지하면 자동으로 재학습 및 지표 업데이트됩니다."
        )
        st.stop()

    # ── 레이턴시 카드 ─────────────────────────────────────────────────────
    st.markdown("### ⚡ API 응답 레이턴시 (p95)")
    lat_col1, lat_col2 = st.columns(2)

    search_lat  = m["search"]["latency_ms"]
    rec_lat     = m["recommend"]["latency_ms"]

    lat_col1.metric(
        "검색 API 레이턴시",
        f"{search_lat} ms",
        delta="✅ 기준 충족" if search_lat <= 200 else "❌ 200ms 초과",
        delta_color="normal" if search_lat <= 200 else "inverse",
    )
    lat_col2.metric(
        "추천 API 레이턴시",
        f"{rec_lat} ms",
        delta="✅ 기준 충족" if rec_lat <= 200 else "❌ 200ms 초과",
        delta_color="normal" if rec_lat <= 200 else "inverse",
    )

    st.divider()

    # ── 검색 / 추천 Gauge 차트 ───────────────────────────────────────────
    col_search, col_rec = st.columns(2)

    with col_search:
        st.markdown("#### 🔍 Stage 1: 멀티모달 검색 (CLIP+FAISS)")

        fig_mrr = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=m["search"]["mrr"],
                delta={"reference": 0.55, "valueformat": ".3f"},
                title={"text": "MRR (목표 ≥ 0.55)"},
                gauge={
                    "axis":      {"range": [0, 1]},
                    "bar":       {"color": "#1565C0"},
                    "bgcolor":   "#E3F2FD",
                    "steps": [
                        {"range": [0,    0.55], "color": "#FFCDD2"},
                        {"range": [0.55, 1.0],  "color": "#C8E6C9"},
                    ],
                    "threshold": {
                        "line":  {"color": "red", "width": 3},
                        "value": 0.55,
                    },
                },
            )
        )
        fig_mrr.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_mrr, width='stretch')

        ndcg10 = m["search"]["ndcg_10"]
        ndcg_color = "🟢" if ndcg10 >= 0.50 else "🔴"
        st.markdown(
            f"{ndcg_color} **NDCG@10:** `{ndcg10:.4f}`  (목표 ≥ 0.50)"
        )

    with col_rec:
        st.markdown("#### 🎁 Stage 2-3: 랭킹 및 추천 (DeepFM)")

        fig_hr = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=m["recommend"]["hitrate_50"],
                delta={"reference": 0.20, "valueformat": ".3f"},
                title={"text": "HitRate@50 (목표 ≥ 0.20)"},
                gauge={
                    "axis":      {"range": [0, 0.5]},
                    "bar":       {"color": "#2E7D32"},
                    "bgcolor":   "#E8F5E9",
                    "steps": [
                        {"range": [0,   0.20], "color": "#FFCDD2"},
                        {"range": [0.20, 0.5], "color": "#C8E6C9"},
                    ],
                    "threshold": {
                        "line":  {"color": "red", "width": 3},
                        "value": 0.20,
                    },
                },
            )
        )
        fig_hr.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_hr, width='stretch')  # 🚨 [수정됨] fig_mrr 오타를 fig_hr로 고쳤습니다!

        r300  = m["recommend"]["recall_300"]
        auc   = m["recommend"]["auc"]
        cov   = m["recommend"]["coverage"]
        ndcg50 = m["recommend"].get("ndcg_50", 0.0)

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        r300_c  = "🟢" if r300  >= 0.30 else "🔴"
        auc_c   = "🟢" if auc   >= 0.70 else "🔴"
        cov_c   = "🟢" if cov   >= 0.20 else "🔴"
        ndcg_c  = "🟢" if ndcg50 >= 0.08 else "🔴"

        m_c1.metric("Recall@300",  f"{r300_c} {r300:.3f}",  help="목표 ≥ 0.30")
        m_c2.metric("DeepFM AUC",  f"{auc_c} {auc:.3f}",   help="목표 ≥ 0.70")
        m_c3.metric("Coverage",    f"{cov_c} {cov:.3f}",   help="목표 ≥ 0.20")
        m_c4.metric("NDCG@50",     f"{ndcg_c} {ndcg50:.3f}", help="목표 ≥ 0.08")

    st.divider()

    # ── 종합 레이더 차트 (정규화) ────────────────────────────────────────
    st.markdown("### 📡 종합 성능 레이더 차트")

    # 각 지표를 목표값 대비 달성률(0~1)로 정규화
    radar_labels = [
        "MRR\n(≥0.55)", "NDCG@10\n(≥0.50)", "Recall@300\n(≥0.30)",
        "AUC\n(≥0.70)", "HitRate@50\n(≥0.20)", "Coverage\n(≥0.20)",
    ]
    radar_actual = [
        m["search"]["mrr"]              / 0.55,
        m["search"]["ndcg_10"]          / 0.50,
        m["recommend"]["recall_300"]    / 0.30,
        m["recommend"]["auc"]           / 0.70,
        m["recommend"]["hitrate_50"]    / 0.20,
        m["recommend"]["coverage"]      / 0.20,
    ]
    # 1.0 을 초과해도 최대 1.5로 클리핑해서 표시
    radar_actual = [min(v, 1.5) for v in radar_actual]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_actual + [radar_actual[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(33,150,243,0.25)",
            line=dict(color="#1565C0", width=2),
            name="달성률 (목표=1.0)",
        )
    )
    # 목표 기준선
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[1.0] * (len(radar_labels) + 1),
            theta=radar_labels + [radar_labels[0]],
            line=dict(color="red", width=1.5, dash="dash"),
            name="목표 기준 (1.0)",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1.5], tickfont=dict(size=10))),
        showlegend=True,
        height=420,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_radar, width='stretch')

    st.divider()

    # ── 명세서 준수 체크리스트 ──────────────────────────────────────────
    st.markdown("### ✅ 명세서 요구사항 준수 검증")

    checks = {
        "MRR ≥ 0.55":         m["search"]["mrr"]           >= 0.55,
        "NDCG@10 ≥ 0.50":     m["search"]["ndcg_10"]       >= 0.50,
        "Recall@300 ≥ 0.30":  m["recommend"]["recall_300"] >= 0.30,
        "DeepFM AUC ≥ 0.70":  m["recommend"]["auc"]        >= 0.70,
        "HitRate@50 ≥ 0.20":  m["recommend"]["hitrate_50"] >= 0.20,
        "NDCG@50 ≥ 0.08":     m["recommend"].get("ndcg_50", 0) >= 0.08,
        "Coverage ≥ 0.20":    m["recommend"]["coverage"]   >= 0.20,
        "Latency ≤ 200ms":    max(
                                  m["search"]["latency_ms"],
                                  m["recommend"]["latency_ms"]
                              ) <= 200,
    }

    passed = sum(checks.values())
    total  = len(checks)
    st.progress(passed / total, text=f"**{passed}/{total}** 항목 충족")

    check_cols = st.columns(4)
    for i, (label, ok) in enumerate(checks.items()):
        check_cols[i % 4].markdown(
            f"{'✅' if ok else '❌'} **{label}**"
        )

    if st.button("🔄 지표 새로고침", key="refresh_metrics"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 ─ A/B 테스트 엔진
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔬 A/B 테스트: 기존 인기도 추천(A) vs DeepFM 개인화 추천(B)")
    st.markdown("두 그룹의 구매 전환율(CVR) 차이를 **Chi-square 검정**으로 통계적 유의성을 검증합니다.")

    # ── 현재 실측 CVR 자동 채우기 힌트 ──────────────────────────────────
    auto_cvr_a, auto_cvr_b = 3.2, 4.5  # 기본값

    def _safe_cvr(purchase_count: int, total_count: int) -> float:
        """CVR 계산 + 0~99.9% 안전 범위로 클리핑."""
        if total_count <= 0:
            return 0.1
        cvr = purchase_count / total_count * 100
        # number_input 의 max_value(99.9)를 넘지 않도록 클리핑
        return float(max(0.1, min(99.9, round(cvr, 2))))

    if not df_logs.empty:
        evt = df_logs["event_type"].value_counts()
        v_  = int(evt.get("view",     0))
        p_  = int(evt.get("purchase", 0))
        # 분모를 view 가 아닌 전체 세션(view+cart+purchase) 으로 사용
        # → view 보다 purchase 가 많아도 100% 를 안 넘김
        total_evt_b = v_ + int(evt.get("cart", 0)) + p_
        auto_cvr_b = _safe_cvr(p_, total_evt_b)

    archive_res = load_latest_archive_logs()
    if isinstance(archive_res, tuple) and not archive_res[0].empty:
        df_arch = archive_res[0]
        av_ = len(df_arch)  # 전체 이벤트 수
        ap_ = int((df_arch["event_type"] == "purchase").sum())
        auto_cvr_a = _safe_cvr(ap_, av_)

    st.info(
        f"💡 **자동 채우기:** "
        f"A그룹(과거 archive 기준) ≈ **{auto_cvr_a}%**, "
        f"B그룹(현재 실시간 기준) ≈ **{auto_cvr_b}%** "
        f"— 분모: 전체 이벤트(view+cart+purchase). 입력값을 수정하거나 그대로 검정을 실행하세요."
    )

    # ── 입력 위젯 ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        sample_size = st.number_input(
            "실험 유저 수 (N, 그룹당)", 1000, 200_000, 10_000, 1000
        )
    with c2:
        cvr_A_pct = st.number_input(
            "A그룹 전환율 (%)", 0.1, 99.9, float(auto_cvr_a), 0.1
        )
    with c3:
        cvr_B_pct = st.number_input(
            "B그룹 전환율 (%)", 0.1, 99.9, float(auto_cvr_b), 0.1
        )

    if st.button("🚀 통계 검정 실행", type="primary", width='stretch'):
        p_A = cvr_A_pct / 100
        p_B = cvr_B_pct / 100

        conv_A = int(sample_size * p_A)
        conv_B = int(sample_size * p_B)

        # ── Chi-square 검정 ──────────────────────────────────────────
        contingency = [
            [conv_A,   sample_size - conv_A],
            [conv_B,   sample_size - conv_B],
        ]
        chi2_stat, p_value, dof, _ = chi2_contingency(contingency)

        # ── 95% 신뢰구간 (정규 근사) ────────────────────────────────
        z_alpha = norm.ppf(0.975)  # ≈1.96
        se      = np.sqrt(p_A * (1 - p_A) / sample_size + p_B * (1 - p_B) / sample_size)
        diff    = p_B - p_A
        ci_low  = diff - z_alpha * se
        ci_high = diff + z_alpha * se
        lift    = (p_B / p_A - 1) * 100 if p_A > 0 else 0.0

        st.divider()
        st.markdown("### 📊 검정 결과")

        # ── 핵심 지표 카드 ──────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("A그룹 CVR", f"{p_A*100:.2f}%", f"{conv_A:,}명 전환")
        r2.metric("B그룹 CVR", f"{p_B*100:.2f}%", f"{conv_B:,}명 전환")
        r3.metric("Lift (B vs A)", f"{lift:+.1f}%")
        r4.metric("p-value", f"{p_value:.5f}", "✅ 유의" if p_value < 0.05 else "❌ 비유의")

        # ── 통계적 유의성 배너 ──────────────────────────────────────
        if p_value < 0.05:
            st.success(
                f"🎉 **통계적으로 유의미한 차이 확인 (p={p_value:.5f} < 0.05)**\n\n"
                f"DeepFM 개인화 추천(B그룹)의 전환율이 우연이 아닌 것으로 검증되었습니다. "
                f"신뢰구간 내 Lift: **{ci_low*100:+.2f}% ~ {ci_high*100:+.2f}%**"
            )
        else:
            st.warning(
                f"⚠️ **통계적 유의성 미확보 (p={p_value:.5f} ≥ 0.05)**\n\n"
                f"두 그룹 차이가 우연일 가능성이 있습니다. "
                f"실험 기간을 연장하거나 표본 수를 늘려보세요. "
                f"현재 신뢰구간: **{ci_low*100:+.2f}% ~ {ci_high*100:+.2f}%**"
            )

        # ── 세부 통계 정보 ──────────────────────────────────────────
        with st.expander("🔬 세부 통계 수치 보기"):
            st.markdown(
                f"""
| 항목 | 값 |
|------|-----|
| Chi-square 통계량 | `{chi2_stat:.4f}` |
| 자유도(dof) | `{dof}` |
| p-value | `{p_value:.6f}` |
| 전환율 차이 (B−A) | `{diff*100:+.3f}%` |
| **95% 신뢰구간** | **`[{ci_low*100:+.3f}%, {ci_high*100:+.3f}%]`** |
| 표준오차 | `{se*100:.4f}%` |
| 표본 크기 (그룹당) | `{sample_size:,}` |
                """
            )

        # ── 전환율 비교 바 차트 ──────────────────────────────────────
        fig_ab = px.bar(
            x=["A 그룹 (기존 인기도 추천)", "B 그룹 (DeepFM 개인화 추천)"],
            y=[p_A * 100, p_B * 100],
            color=["A 그룹", "B 그룹"],
            color_discrete_map={"A 그룹": "#9E9E9E", "B 그룹": "#FF7043"},
            text=[f"{p_A*100:.2f}%", f"{p_B*100:.2f}%"],
            error_y=[z_alpha * np.sqrt(p_A*(1-p_A)/sample_size)*100,
                     z_alpha * np.sqrt(p_B*(1-p_B)/sample_size)*100],
            labels={"x": "실험 그룹", "y": "구매 전환율 (CVR) %"},
            title="그룹별 CVR 비교 (오차막대 = 95% 신뢰구간)",
        )
        fig_ab.update_traces(textposition="outside")
        fig_ab.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig_ab, width='stretch')

        # ── 표본 크기 파워 분석 (필요 표본 수 계산) ────────────────
        st.divider()
        st.markdown("### 📐 통계적 검정력(Power) 분석")

        # 최소 표본 수 (검정력 80%, α=0.05, 단측)
        if p_A > 0 and p_B > p_A:
            p_pool    = (p_A + p_B) / 2
            z_beta    = norm.ppf(0.80)  # 80% power
            numerator = (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool))
                         + z_beta  * np.sqrt(p_A*(1-p_A) + p_B*(1-p_B))) ** 2
            min_n     = int(np.ceil(numerator / (p_B - p_A) ** 2))

            pow_msg = (
                f"현재 차이({diff*100:+.2f}%)를 p<0.05 수준에서 검출하려면 "
                f"그룹당 최소 **{min_n:,}명**이 필요합니다 (검정력 80% 기준)."
            )
            if sample_size >= min_n:
                st.success(f"✅ 현재 표본 수({sample_size:,})는 충분합니다. {pow_msg}")
            else:
                st.warning(f"⚠️ 표본이 부족합니다. {pow_msg}")
        else:
            st.info("B그룹 전환율이 A그룹보다 높을 때 검정력 분석이 표시됩니다.")