import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 웹페이지 기본 설정
st.set_page_config(
    page_title="대한민국 SVIR 바이러스-백신 시뮬레이터",
    page_icon="🦠",
    layout="wide"
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
    "COVID-19 (원형)": {"R0": 2.5, "gamma": 1/14, "desc": "초기 코로나19 바이러스 (기초감염재생산수 R0 = 2.5)"},
    "COVID-19 (델타 변이)": {"R0": 5.0, "gamma": 1/10, "desc": "높은 전파력을 가진 델타 변이 (R0 = 5.0)"},
    "COVID-19 (오미크론 변이)": {"R0": 10.0, "gamma": 1/7, "desc": "극도로 빠른 전파력의 오미크론 변이 (R0 = 10.0)"},
    "SARS": {"R0": 3.0, "gamma": 1/12, "desc": "사스 중증급성호흡기증후군 (R0 = 3.0)"}
}

VACCINES = {
    "mRNA 백신": {"efficacy": 0.95, "desc": "화이자/모더나 등 높은 예방 효과 (효능 95%)"},
    "재조합 백신 (합성항원)": {"efficacy": 0.80, "desc": "노바백스 등 안정적인 면역 반응 (효능 80%)"},
    "생백신 (약독화)": {"efficacy": 0.70, "desc": "전통적 방식의 약독화 백신 (효능 70%)"}
}

# 2. SVIR 미분방정식
def svir_model(y, t, N, beta, gamma, v, efficacy):
    S, V, I, R = y
    beta_v = beta * (1.0 - efficacy)
    
    dSdt = -beta * S * I / N - v * S
    dVdt = v * S - beta_v * V * I / N
    dIdt = beta * S * I / N + beta_v * V * I / N - gamma * I
    dRdt = gamma * I
    
    return [dSdt, dVdt, dIdt, dRdt]

# 3. 좌측 사이드바 조작창
st.sidebar.header("⚙️ 시뮬레이션 파라미터 설정")

selected_virus = st.sidebar.selectbox("1. 바이러스 종류 선택", list(VIRUSES.keys()), index=2)
selected_vaccine = st.sidebar.selectbox("2. 백신 종류 선택", list(VACCINES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 시뮬레이션 환경 조건")
N_pop = st.sidebar.number_input("대한민국 인구수 (명)", value=51600000, step=1000000, format="%d")
initial_infected = st.sidebar.number_input("초기 감염자 수 (명)", value=100, step=10)
daily_vac_rate_pct = st.sidebar.slider("일일 백신 접종률 (%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1) / 100.0
sim_days = st.sidebar.slider("시뮬레이션 기간 (일)", min_value=30, max_value=365, value=180, step=10)

# 선택한 조건 안내 표시
st.info(f"**선택된 바이러스**: {selected_virus} (*{VIRUSES[selected_virus]['desc']}*)\n\n**선택된 백신**: {selected_vaccine} (*{VACCINES[selected_vaccine]['desc']}*)")

# 4. 시뮬레이션 수치 계산
virus_info = VIRUSES[selected_virus]
vac_info = VACCINES[selected_vaccine]

gamma = virus_info["gamma"]
R0_val = virus_info["R0"]
beta = R0_val * gamma
efficacy = vac_info["efficacy"]

I0 = initial_infected
S0 = N_pop - I0
V0 = 0
R0_pop = 0

t = np.linspace(0, sim_days, sim_days)
solution = odeint(svir_model, [S0, V0, I0, R0_pop], t, args=(N_pop, beta, gamma, daily_vac_rate_pct, efficacy))
S, V, I, R = solution.T

# 주요 지표 요약
max_infected = int(np.max(I))
peak_day = int(np.argmax(I))
total_vaccinated = int(V[-1])
total_recovered = int(R[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("최대 동시 감염자", f"{max_infected:,} 명")
col2.metric("감염 피크(정점)", f"{peak_day} 일째")
col3.metric("최종 백신 접종자", f"{total_vaccinated:,} 명")
col4.metric("최종 회복/면역자", f"{total_recovered:,} 명")

st.markdown("---")

# 5. 그래프 시각화
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, S / 1e6, label='S (감염 가능 미접종자)', color='#1f77b4', linewidth=2)
ax.plot(t, V / 1e6, label='V (백신 접종 완료자)', color='#2ca02c', linewidth=2)
ax.plot(t, I / 1e6, label='I (현재 감염자)', color='#d62728', linewidth=2)
ax.plot(t, R / 1e6, label='R (회복자/면역)', color='#7f7f7f', linewidth=2, linestyle='--')

ax.set_title(f"SVIR 시뮬레이션 - {selected_virus} & {selected_vaccine}", fontsize=14, pad=12)
ax.set_xlabel("기간 (일)", fontsize=11)
ax.set_ylabel("인구수 (백만 명)", fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=10, loc='center right')

st.pyplot(fig)
