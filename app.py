import numpy as np
import pandas as pd
import streamlit as st

# 웹페이지 기본 설정
st.set_page_config(
    page_title="대한민국 SVIR 바이러스-백신 시뮬레이터",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 대한민국 SVIR 감염병 - 백신 시뮬레이터")
st.markdown("""
이 시뮬레이터는 **SVIR 모델(Susceptible-Vaccinated-Infected-Recovered)**을 사용하여 바이러스 변이 및 백신 종류에 따른 감염 확산 양상을 시각화합니다.
""")

# 1. 바이러스 및 백신 매개변수 데이터 정의
VIRUSES = {
    "COVID-19 (원형)": {
        "R0": 2.5,
        "gamma": 1 / 14,
        "desc": "초기 코로나19 바이러스 (기초감염재생산수 R0 = 2.5)",
    },
    "COVID-19 (델타 변이)": {
        "R0": 5.0,
        "gamma": 1 / 10,
        "desc": "높은 전파력을 가진 델타 변이 (R0 = 5.0)",
    },
    "COVID-19 (오미크론 변이)": {
        "R0": 10.0,
        "gamma": 1 / 7,
        "desc": "극도로 빠른 전파력의 오미크론 변이 (R0 = 10.0)",
    },
    "SARS": {
        "R0": 3.0,
        "gamma": 1 / 12,
        "desc": "사스 중증급성호흡기증후군 (R0 = 3.0)",
    },
}

VACCINES = {
    "mRNA 백신": {
        "efficacy": 0.95,
        "desc": "화이자/모더나 등 높은 예방 효과 (효능 95%)",
    },
    "생백신 (약독화)": {
        "efficacy": 0.70,
        "desc": "전통적 방식의 약독화 백신 (효능 70%)",
    },
    "사백신 (불활성화)": {
        "efficacy": 0.60,
        "desc": "사멸시킨 바이러스 활용 백신, 예: 시노팜/시노백 (효능 60%)",
    },
}


# 2. SVIR 수치해석 모델 연산 함수
def solve_svir_fast(N, beta, gamma, v, efficacy, I0, days):
  dt = 1.0
  steps = int(days)

  t_arr = np.linspace(0, days, steps)
  S = np.zeros(steps)
  V = np.zeros(steps)
  I = np.zeros(steps)
  R = np.zeros(steps)

  I[0] = I0
  S[0] = N - I0
  V[0] = 0
  R[0] = 0

  beta_v = beta * (1.0 - efficacy)

  for i in range(steps - 1):
    s_curr, v_curr, i_curr = S[i], V[i], I[i]

    dS = (-beta * s_curr * i_curr / N - v * s_curr) * dt
    dV = (v * s_curr - beta_v * v_curr * i_curr / N) * dt
    dI = (
        beta * s_curr * i_curr / N
        + beta_v * v_curr * i_curr / N
        - gamma * i_curr
    ) * dt
    dR = (gamma * i_curr) * dt

    S[i + 1] = max(0, s_curr + dS)
    V[i + 1] = max(0, v_curr + dV)
    I[i + 1] = max(0, i_curr + dI)
    R[i + 1] = max(0, R[i] + dR)

  return t_arr, S, V, I, R


# 3. 사이드바 조작창 및 지표 가이드
st.sidebar.header("⚙️ 시뮬레이션 파라미터 설정")

selected_virus = st.sidebar.selectbox(
    "1. 바이러스 종류 선택", list(VIRUSES.keys()), index=0
)
selected_vaccine = st.sidebar.selectbox(
    "2. 백신 종류 선택", list(VACCINES.keys()), index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 시뮬레이션 환경 조건")
N_pop = st.sidebar.number_input(
    "대한민국 인구수 (명)", value=51600000, step=1000000, format="%d"
)
initial_infected = st.sidebar.number_input(
    "초기 감염자 수 (명)", value=100, step=10
)
daily_vac_rate_pct = (
    st.sidebar.slider(
        "일일 백신 접종률 (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )
    / 100.0
)
sim_days = st.sidebar.slider(
    "시뮬레이션 기간 (일)", min_value=30, max_value=365, value=180, step=10
)

# 사이드바 하단에 지표 설명 가이드 배치
st.sidebar.markdown("---")
st.sidebar.subheader("📖 지표 가이드")
st.sidebar.caption("""
- **S (Susceptible)**: 감염 가능 미접종자
- **V (Vaccinated)**: 백신 접종 완료자
- **I (Infected)**: 현재 감염자
- **R (Recovered)**: 회복자 및 면역 확보자
- **전파 피크(억제 최소)**: 감염자 수가 최대에 달하여 전파 억제력이 가장 무너진 시점
- **🔥 최대 전파 억제 전환점**: 백신/면역 효과로 바이러스 감소폭이 가장 커지며 전파 억제가 극대화되는 시점
""")

# 4. 모델 연산 실행
virus_info = VIRUSES[selected_virus]
vac_info = VACCINES[selected_vaccine]

gamma = virus_info["gamma"]
R0_val = virus_info["R0"]
beta = R0_val * gamma
efficacy = vac_info["efficacy"]

t_arr, S, V, I, R = solve_svir_fast(
    N=N_pop,
    beta=beta,
    gamma=gamma,
    v=daily_vac_rate_pct,
    efficacy=efficacy,
    I0=initial_infected,
    days=sim_days,
)

# 5. 전파 억제 관련 지표 분석
max_infected = int(np.max(I))
peak_idx = np.argmax(I)
peak_day = int(t_arr[peak_idx])

# 전파 억제 전환점 계산
dI = np.diff(I)
suppression_idx = np.argmin(dI) + 1 if len(dI) > 0 else peak_idx
suppression_day = int(t_arr[suppression_idx])

# 주요 지표 상단 표시
st.info(
    f"**선택된 바이러스**: {selected_virus} | **선택된 백신**:"
    f" {selected_vaccine}"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("최대 동시 감염자", f"{max_infected:,} 명")
col2.metric(
    "전파 피크 (억제 최소)",
    f"{peak_day} 일째",
    help="감염자 수가 최고조에 달한 지점입니다.",
)
col3.metric(
    "🔥 최대 전파 억제 전환점",
    f"{suppression_day} 일째",
    help="감염 감소 속도가 가장 빠른 전환점입니다.",
)
col4.metric("최종 백신 접종자", f"{int(V[-1]):,} 명")

st.markdown("---")

# 6. 시각화 그래프
st.subheader("📊 시뮬레이션 결과 그래프 (단위: 백만 명)")

chart_df = pd.DataFrame(
    {
        "S (감염 가능 미접종자)": S / 1e6,
        "V (백신 접종 완료자)": V / 1e6,
        "I (현재 감염자)": I / 1e6,
        "R (회복자/면역)": R / 1e6,
    },
    index=t_arr,
)

st.line_chart(chart_df, height=450)
