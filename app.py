import streamlit as st
import numpy as np

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

# 1. 바이러스 및 백신 매개변수 데이터 정의 (메르스 추가)
VIRUSES = {
    "COVID-19 (원형)": {"R0": 2.5, "gamma": 1/14, "desc": "초기 코로나19 바이러스 (기초감염재생산수 R0 = 2.5)"},
    "COVID-19 (델타 변이)": {"R0": 5.0, "gamma": 1/10, "desc": "높은 전파력을 가진 델타 변이 (R0 = 5.0)"},
    "COVID-19 (오미크론 변이)": {"R0": 10.0, "gamma": 1/7, "desc": "극도로 빠른 전파력의 오미크론 변이 (R0 = 10.0)"},
    "SARS": {"R0": 3.0, "gamma": 1/12, "desc": "사스 중증급성호흡기증후군 (R0 = 3.0)"},
    "MERS (메르스)": {"R0": 0.8, "gamma": 1/14, "desc": "중동호흡기증후군 (R0 = 0.8, 치사율이 높고 지역적 전파 특성)"}
}

# 백신 종류 설정
VACCINES = {
    "mRNA 백신": {"efficacy": 0.95, "desc": "화이자/모더나 등 높은 예방 효과 (효능 95%)"},
    "생백신 (약독화)": {"efficacy": 0.70, "desc": "전통적 방식의 약독화 백신 (효능 70%)"},
    "사백신 (불활성화)": {"efficacy": 0.60, "desc": "사멸시킨 바이러스 활용 백신, 예: 시노팜/시노백 (효능 60%)"}
}

# 2. 최적화된 수치해석 함수 (연산 렉 제거)
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
        dI = (beta * s_curr * i_curr / N + beta_v * v_curr * i_curr / N - gamma * i_curr) * dt
        dR = (gamma * i_curr) * dt
        
        S[i+1] = max(0, s_curr + dS)
        V[i+1] = max(0, v_curr + dV)
        I[i+1] = max(0, i_curr + dI)
        R[i+1] = max(0, R[i] + dR)
        
    return t_arr, S, V, I, R

# 3. 사이드바 조작창
st.sidebar.header("⚙️ 시뮬레이션 파라미터 설정")

selected_virus = st.sidebar.selectbox("1. 바이러스 종류 선택", list(VIRUSES.keys()), index=2)
selected_vaccine = st.sidebar.selectbox("2. 백신 종류 선택", list(VACCINES.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 시뮬레이션 환경 조건")
N_pop = st.sidebar.number_input("대한민국 인구수 (명)", value=51600000, step=1000000, format="%d")
initial_infected = st.sidebar.number_input("초기 감염자 수 (명)", value=100, step=10)
daily_vac_rate_pct = st.sidebar.slider("일일 백신 접종률 (%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1) / 100.0
sim_days = st.sidebar.slider("시뮬레이션 기간 (일)", min_value=30, max_value=365, value=180, step=10)

st.info(f"**선택된 바이러스**: {selected_virus} (*{VIRUSES[selected_virus]['desc']}*)\n\n**선택된 백신**: {selected_vaccine} (*{VACCINES[selected_vaccine]['desc']}*)")

# 4. 모델 연산 실행
virus_info = VIRUSES[selected_virus]
vac_info = VACCINES[selected_vaccine]

gamma = virus_info["gamma"]
R0_val = virus_info["R0"]
beta = R0_val * gamma
efficacy = vac_info["efficacy"]

# 속도 최적화 함수 사용
t_arr, S, V, I, R = solve_svir_fast(
    N=N_pop, 
    beta=beta, 
    gamma=gamma, 
    v=daily_vac_rate_pct, 
    efficacy=efficacy, 
    I0=initial_infected, 
    days=sim_days
)

# 주요 지표 요약
max_infected = int(np.max(I))
peak_idx = np.argmax(I)
peak_day = int(t_arr[peak_idx])
total_vaccinated = int(V[-1])
total_recovered = int(R[-1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("최대 동시 감염자", f"{max_infected:,} 명")
col2.metric("감염 피크(정점)", f"{peak_day} 일째")
col3.metric("최종 백신 접종자", f"{total_vaccinated:,} 명")
col4.metric("최종 회복/면역자", f"{total_recovered:,} 명")

st.markdown("---")

# 5. 시각화 그래프
st.subheader("📊 시뮬레이션 결과 그래프 (단위: 백만 명)")

chart_data = {
    "S (감염 가능 미접종자)": S / 1e6,
    "V (백신 접종 완료자)": V / 1e6,
    "I (현재 감염자)": I / 1e6,
    "R (회복자/면역)": R / 1e6
}

st.line_chart(chart_data, height=400)

st.markdown("---")

# 6. 수학적 일반화 (Mathematical Generalization) 설명
st.subheader("📐 SVIR 모델의 수학적 일반화 (Mathematical Generalization)")

st.markdown(r"""
전체 인구수 $N$이 일정하다고 가정할 때 ($N = S(t) + V(t) + I(t) + R(t)$), 시간 $t$에 따른 상태 변화는 다음과 같은 **비선형 연립 미분방정식(Nonlinear Ordinary Differential Equations)**으로 정의됩니다.

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta \frac{S I}{N} - v S \\[8pt]
\frac{dV}{dt} &= v S - (1 - e)\beta \frac{V I}{N} \\[8pt]
\frac{dI}{dt} &= \beta \frac{S I}{N} + (1 - e)\beta \frac{V I}{N} - \gamma I \\[8pt]
\frac{dR}{dt} &= \gamma I
\end{aligned}
$$
""")

with st.expander("🔍 수학적 파라미터 및 변수 상세 정의 보기"):
    st.markdown(r"""
    * **상태 변수 (State Variables)**:
      * $S(t)$: 시간 $t$에서의 감염 가능 미접종 인구수 (Susceptible)
      * $V(t)$: 시간 $t$에서의 백신 접종 완료 인구수 (Vaccinated)
      * $I(t)$: 시간 $t$에서의 감염자 인구수 (Infected)
      * $R(t)$: 시간 $t$에서의 회복자 및 면역 형성 인구수 (Recovered)

    * **파라미터 (Parameters)**:
      * $\beta$ (**감염 전파율**, Transmission Rate): $\beta = R_0 \cdot \gamma$
      * $R_0$ (**기초감염재생산수**, Basic Reproduction Number): 감염자 1명이 면역이 없는 집단에서 감염시킬 수 있는 평균 인원수
      * $\gamma$ (**회복율/격리해제율**, Recovery Rate): $\gamma = \frac{1}{\text{감염 유효 기간(일)}}$
      * $v$ (**일일 백신 접종률**, Vaccination Rate): 미접종군($S$)이 매일 백신 접종군($V$)으로 이동하는 비율
      * $e$ (**백신 효능**, Vaccine Efficacy): 감염 방지율 ($0 \le e \le 1$). 미접종군의 감염율이 $\beta$일 때, 접종군의 돌파감염율은 $(1-e)\beta$가 됨
    """)
