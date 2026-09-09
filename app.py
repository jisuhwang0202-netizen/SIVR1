import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
- **S (Susceptible)**: 감염 가능 미접종자
- **V (Vaccinated)**: 백신 접종 완료자
- **I (Infected)**: 현재 감염자
- **R (Recovered)**: 회복자 및 면역 확보자
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


# 3. 사이드바 조작창
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

st.info(
    f"**선택된 바이러스**: {selected_virus}"
    f" (*{VIRUSES[selected_virus]['desc']}*)\n\n**선택된 백신**:"
    f" {selected_vaccine} (*{VACCINES[selected_vaccine]['desc']}*)"
)

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
peak_idx = np.argmax(I)  # 전파 억제가 가장 안 된 정점 (Peak)
peak_day = int(t_arr[peak_idx])

# 전파 억제율이 가장 높아지는 시점 (감염 피크 이후 확산세가 꺾이고 바이러스 전파가 감소하는 전환 지점)
# 피크 이후 일일 감염 감소폭(-dI/dt)이 가장 큰 지점 계산
dI = np.diff(I)
suppression_idx = (
    np.argmin(dI) + 1 if len(dI) > 0 else peak_idx
)  # 감염자 수 감소 속도가 가장 빠른 지점
suppression_day = int(t_arr[suppression_idx])
suppression_val_million = I[suppression_idx] / 1e6

col1, col2, col3, col4 = st.columns(4)
col1.metric("최대 동시 감염자", f"{max_infected:,} 명")
col2.metric("전파 피크(억제 최소)", f"{peak_day} 일째")
col3.metric("🔥 최대 전파 억제 전환점", f"{suppression_day} 일째")
col4.metric("최종 백신 접종자", f"{int(V[-1]):,} 명")

st.markdown("---")

# 6. 시각화 그래프 (Plotly 사용)
st.subheader("📊 시뮬레이션 결과 그래프 (단위: 백만 명)")

fig = go.Figure()

# 기본 곡선 추가
fig.add_trace(
    go.Scatter(
        x=t_arr,
        y=S / 1e6,
        mode="lines",
        name="S (감염 가능 미접종자)",
        line=dict(color="#1f77b4"),
    )
)
fig.add_trace(
    go.Scatter(
        x=t_arr,
        y=V / 1e6,
        mode="lines",
        name="V (백신 접종 완료자)",
        line=dict(color="#2ca02c"),
    )
)
fig.add_trace(
    go.Scatter(
        x=t_arr,
        y=I / 1e6,
        mode="lines",
        name="I (현재 감염자)",
        line=dict(color="#ff7f0e", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=t_arr,
        y=R / 1e6,
        mode="lines",
        name="R (회복자/면역)",
        line=dict(color="#9467bd"),
    )
)

# 🔥 전파 억제력 최고 지점 마커 및 가이드선 추가
fig.add_trace(
    go.Scatter(
        x=[suppression_day],
        y=[suppression_val_million],
        mode="markers+text",
        name="최대 전파 억제 지점",
        marker=dict(size=14, color="red", symbol="star"),
        text=[f"📍 최대 억제 전환점 ({suppression_day}일)"],
        textposition="top right",
    )
)

fig.add_shape(
    type="line",
    x0=suppression_day,
    y0=0,
    x1=suppression_day,
    y1=max(I / 1e6) * 1.1,
    line=dict(color="red", width=1.5, dash="dash"),
)

fig.update_layout(
    xaxis_title="시간 (일)",
    yaxis_title="인구수 (백만 명)",
    hovermode="x unified",
    height=450,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"💡 **붉은색 별표(📍) 지점({suppression_day}일째)**은 백신 접종 효과 및"
    " 집단 면역 형성으로 인해 **바이러스의 감염 감소 폭이 가장 커지며 전파"
    " 억제력이 극대화되는 시점**입니다."
)
