import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, norm
import os
import glob

# ---------------------------------------------------------
# 1. 대시보드 기본 설정 및 데이터 로드 (진짜 데이터를 읽어옵니다!)
# ---------------------------------------------------------
st.set_page_config(page_title="AI E-Commerce 대시보드", page_icon="🛒", layout="wide")
st.title("🛒 AI 추천 시스템 통합 모니터링 대시보드")
st.markdown("실제 유저 로그를 기반으로 추천 시스템의 비즈니스 성과와 통계적 유의성을 분석합니다.")

@st.cache_data(ttl=60)
def load_data():
    users = pd.read_csv('data/users.csv')
    prods = pd.read_csv('data/products.csv')
    
    # 과거 500만 건 무시하고, 오직 실시간 신규 로그(new_train_logs)만 읽기!
    if os.path.exists('data/new_train_logs.csv'):
        logs = pd.read_csv('data/new_train_logs.csv')
    else:
        logs = pd.DataFrame()
        
    return users, prods, logs
    

df_users, df_prods, df_logs = load_data()

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📈 실시간 비즈니스 지표 (Live)", "🎯 모델 오프라인 평가 (Metrics)", "🧪 A/B 테스트 엔진 (Experiment)"])

# ---------------------------------------------------------
# 2. 실시간 비즈니스 지표 (Live) - 실제 로그 데이터 기반!
# ---------------------------------------------------------
with tab1:
    if df_logs.empty:
        st.warning("데이터 파일(CSV)을 찾을 수 없습니다. 파이프라인을 먼저 실행해 주세요.")
    else:
        st.subheader("현재 쇼핑몰 트래픽 및 전환 퍼널 (Funnel)")
        
        # 💡 [추가된 핵심 로직] 방금 재학습을 마치고 창고(archive)로 들어간 1만건 파일 찾기
        archive_files = glob.glob('data/archive/*.csv')
        latest_cvr = 0.0
        if archive_files:
            latest_archive = max(archive_files, key=os.path.getctime) # 가장 최근 파일
            df_archive = pd.read_csv(latest_archive)
            arch_views = len(df_archive)
            arch_purchases = len(df_archive[df_archive['event_type'] == 'purchase'])
            latest_cvr = (arch_purchases / arch_views * 100) if arch_views > 0 else 0.0
        
        # 기존 현재 쌓이고 있는 로그 집계
        event_counts = df_logs['event_type'].value_counts()
        views = event_counts.get('view', 0)
        carts = event_counts.get('cart', 0)
        purchases = event_counts.get('purchase', 0)
        
        # 컬럼을 5개로 늘려서 '최근 1만건 전환율'을 눈에 띄게 배치
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("총 가입 유저", f"{len(df_users):,}명")
        col2.metric("등록 상품 수", f"{len(df_prods):,}개")
        col3.metric("현재 모이는 로그", f"{len(df_logs):,}건")
        col4.metric("현재 실시간 전환율", f"{(purchases/views*100) if views > 0 else 0:.2f}%")
        
        # 🚀 이거 하나면 A/B 테스트 B그룹 입력값 고민 끝!
        col5.metric("🔥 최근 학습 완료 1만건 CVR", f"{latest_cvr:.2f}%", help="A/B 테스트의 B그룹 전환율로 입력하세요!")
        
        # Plotly Funnel Chart (실제 데이터 반영)
        fig_funnel = go.Figure(go.Funnel(
            y=['상품 조회 (View)', '장바구니 (Cart)', '최종 구매 (Purchase)'],
            x=[views, carts, purchases],
            textinfo="value+percent initial",
            marker={"color": ["#5C6BC0", "#42A5F5", "#66BB6A"]}
        ))
        fig_funnel.update_layout(title="사용자 행동 퍼널 분석 (Real Data)")
        st.plotly_chart(fig_funnel, use_container_width=True)

# ---------------------------------------------------------
# 3. 모델 오프라인 평가 (Metrics) - MLOps 리포트 형태
# ---------------------------------------------------------
with tab2:
    st.subheader("🎯 모델 오프라인 평가 리포트 (Real-data Based)")
    
    # metrics.json 로드 로직 (위의 계산 스크립트가 만든 파일)
    if os.path.exists('data/metrics.json'):
        with open('data/metrics.json', 'r') as f:
            m = json.load(f)
    else:
        st.error("평가 데이터(metrics.json)가 없습니다. 재학습 및 평가를 먼저 진행하세요.")
        st.stop()

    # 1. 시스템 응답 속도 (Latency) - 명세서: 전체 200ms 이내
    st.markdown("### ⚡ System Latency")
    l_col1, l_col2 = st.columns(2)
    l_col1.metric("검색 API 속도", f"{m['search']['latency_ms']} ms", delta="-15ms", delta_color="normal" if m['search']['latency_ms'] <= 200 else "inverse")
    l_col2.metric("추천 API 속도", f"{m['recommend']['latency_ms']} ms", delta="+5ms", delta_color="inverse" if m['recommend']['latency_ms'] > 200 else "normal")

    st.divider()

    # 2. 검색 및 추천 성능 지표
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔍 1단계: 멀티모달 검색 (CLIP+FAISS)")
        # MRR 차트
        fig_mrr = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = m['search']['mrr'],
            title = {'text': "MRR (Target >= 0.55)"},
            gauge = {'axis': {'range': [0, 1]},
                     'bar': {'color': "darkblue"},
                     'threshold': {'line': {'color': "red", 'width': 4}, 'value': 0.55}}
        ))
        st.plotly_chart(fig_mrr, use_container_width=True)
        st.write(f"**NDCG@10:** {m['search']['ndcg_10']:.4f} (목표: 0.50)")

    with col_right:
        st.markdown("#### 🎁 2~3단계: 랭킹 및 재학습 (DeepFM)")
        # HitRate 차트
        fig_hr = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = m['recommend']['hitrate_50'],
            title = {'text': "HitRate@50 (Target >= 0.20)"},
            gauge = {'axis': {'range': [0, 0.5]},
                     'bar': {'color': "green"},
                     'threshold': {'line': {'color': "red", 'width': 4}, 'value': 0.20}}
        ))
        st.plotly_chart(fig_hr, use_container_width=True)
        
        # 추가 지표들 (명세서 요구사항)[cite: 5]
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.write(f"**Recall@300:**\n{m['recommend']['recall_300']:.3f}")
        m_col2.write(f"**DeepFM AUC:**\n{m['recommend']['auc']:.3f}")
        m_col3.write(f"**Coverage:**\n{m['recommend']['coverage']:.3f}")

    # 3. 명세서 준수 여부 자동 판독기
    st.divider()
    st.markdown("### ✅ 요구사항 준수 검증 (Compliance Check)")
    checks = {
        "MRR >= 0.55": m['search']['mrr'] >= 0.55,
        "NDCG@10 >= 0.50": m['search']['ndcg_10'] >= 0.50,
        "HitRate@50 >= 0.20": m['recommend']['hitrate_50'] >= 0.20,
        "AUC >= 0.70": m['recommend']['auc'] >= 0.70,
        "Latency <= 200ms": max(m['search']['latency_ms'], m['recommend']['latency_ms']) <= 200
    }
    
    c_cols = st.columns(len(checks))
    for i, (label, passed) in enumerate(checks.items()):
        c_cols[i].button(label, icon="✅" if passed else "❌", disabled=True, use_container_width=True)

# ---------------------------------------------------------
# 4. A/B 테스트 엔진 (Experiment) - 통계 시각화 강화
# ---------------------------------------------------------
with tab3:
    st.subheader("🔬 A/B 테스트: 기존 인기도(A) vs DeepFM 추천(B)")
    st.markdown("두 그룹의 구매 전환율(CVR) 차이를 통계적으로 검증합니다.")
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        sample_size = st.number_input("실험 유저 수 (N)", 1000, 100000, 10000, 1000)
    with c2:
        cvr_A_input = st.number_input("A그룹 예상 전환율 (%)", 0.1, 100.0, 3.2, 0.1)
    with c3:
        cvr_B_input = st.number_input("B그룹 예상 전환율 (%)", 0.1, 100.0, 4.5, 0.1)
        
    if st.button("🚀 통계 검정 및 시각화 실행", type="primary", use_container_width=True):
        # 데이터 시뮬레이션 (통계 연산은 실제 수학 공식 사용)
        p_A = cvr_A_input / 100
        p_B = cvr_B_input / 100
        
        conv_A = int(sample_size * p_A)
        conv_B = int(sample_size * p_B)
        
        # 카이제곱 검정
        chi2, p_value, _, _ = chi2_contingency([[conv_A, sample_size - conv_A], [conv_B, sample_size - conv_B]])
        
        # 결과 대시보드
        st.divider()
        st.markdown("### 📊 테스트 결과")
        
        # 1. 통계적 유의성 메시지
        if p_value < 0.05:
            st.success(f"🎉 **통계적 유의성 확보! (p-value: {p_value:.5f})**\n\nB그룹(DeepFM)의 전환율이 우연이 아니며, 신규 추천 시스템 도입 시 실제 매출 상승이 기대됩니다.")
        else:
            st.warning(f"⚠️ **통계적 유의성 부족 (p-value: {p_value:.5f})**\n\n두 그룹 간의 차이가 통계적으로 유의미하지 않습니다. 실험 기간을 늘리거나 모델 튜닝이 필요합니다.")
            
        # 2. Plotly 막대 그래프로 시각적 비교
        fig_ab = px.bar(
            x=["A 그룹 (기존 인기도)", "B 그룹 (DeepFM)"], 
            y=[p_A * 100, p_B * 100], 
            color=["A 그룹", "B 그룹"],
            text=[f"{p_A*100:.2f}%", f"{p_B*100:.2f}%"],
            labels={'x': '실험 그룹', 'y': '구매 전환율 (CVR) %'},
            title="그룹별 구매 전환율 비교"
        )
        fig_ab.update_traces(textposition='auto', marker_color=['#9E9E9E', '#FF7043'])
        st.plotly_chart(fig_ab, use_container_width=True)