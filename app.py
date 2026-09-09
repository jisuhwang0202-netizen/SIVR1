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

# 1. 바이러스 정의 (전파력 R0, 회복기간) - MERS 제거, B형 간염 추가
VIRUSES = {
    "COVID-19 (원형)": {"R0": 2.5, "gamma": 1/14, "desc": "초기 코로나19 바이러스 (기초감염재생산수 R0 = 2.5)"},
    "COVID-19 (델타 변이)": {"R0": 5.0, "gamma": 1/10, "desc": "높은 전파력을 가진 델타 변이 (R0 = 5.0)"},
    "COVID-19 (오미크론 변이)": {"R0": 10.0, "gamma": 1/7, "desc": "극도로 빠른 전파력의 오미크론 변이 (R0 = 10.0)"},
    "SARS": {"R0": 3.0, "gamma": 1/12, "desc": "사스 중증급성호흡기증후군 (R0 = 3.0)"},
    "B형 간염 (Hepatitis B)": {"R0": 1.8, "gamma": 1/30, "desc": "혈액/체액 매개 간염 바이러스 (R0 = 1.8, 긴 감염 유지 기간)"}
}

# 2. 백신 데이터 및 바이러스별 실제 효능 매트릭스 (Virus-Vaccine Efficacy Matrix)
VACCINE_EFFICACY_MATRIX = {
    "mRNA 백신": {
        "COVID-19 (원형)": 0.95,
        "COVID-19 (델타 변이)": 0.85,
        "COVID-19 (오미크론 변이)": 0.55,
        "SARS": 0.60,
        "B형 간염 (Hepatitis B)": 0.85,
        "desc": "높은 항체 형성률 (변이 바이러스 대응력 우수)"
    },
    "생백신 (약독화)": {
        "COVID-19 (원형)": 0.70,
        "COVID-19 (델타 변이)": 0.60,
        "COVID-19 (오미크론 변이)": 0.40,
        "SARS": 0.65,
        "B형 간염 (Hepatitis B)": 0.70,
        "desc": "전통적 약독화 방식 백신"
    },
    "사백신 (불활성화)": {
        "COVID-19 (원형)": 0.60,
        "COVID-19 (델타 변이)": 0.45,
        "COVID-19 (오미크론 변이)": 0.25,
        "SARS": 0.50,
        "B형 간염 (Hepatitis B)": 0.95, # B형 간염 백신(유전자재조합/사백신)은 예방률 95% 이상으로 우수함
        "desc": "불활성화/항원 기반 백신 (B형 간염 등에 높은 표준 예방율)"
    }
}

# 3. 최적화된 SVIR 수치해석 함수
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
    
    # 돌파 감염률 적용
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

# 4. 사이드바 조작창
st.sidebar.header("⚙️ 시뮬레이션 파라미터 설정")

selected_virus = st.sidebar.selectbox("1. 바이러스 종류 선택", list(VIRUSES.keys()), index=4) # B형 간염 기본 선택
selected_vaccine = st.sidebar.selectbox("2. 백신 종류 선택", list(VACCINE_EFFICACY_MATRIX.keys()), index=2) # 사백신 기본 선택

st.sidebar.markdown("---")
st.sidebar.subheader("📈 시뮬레이션 환경 조건")
N_pop = st.sidebar.number_input("대한민국 인구수 (명)", value=51600000, step=1000000, format="%d")
initial_infected = st.sidebar.number_input("초기 감염자 수 (명)", value=100, step=10)
daily_vac_rate_pct = st.sidebar.slider("일일 백신 접종률 (%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1) / 100.0
sim_days = st.sidebar.slider("시뮬레이션 기간 (일)", min_value=30, max_value=365, value=180, step=10)

# 선택한 바이러스와 백신의 실제 효능 가져오기
virus_info = VIRUSES[selected_virus]
vaccine_info = VACCINE_EFFICACY_MATRIX[selected_vaccine]
actual_efficacy = vaccine_info[selected_virus]

st.info(f"**선택된 바이러스**: {selected_virus} (*{virus_info['desc']}*)\n\n"
        f"**선택된 백신**: {selected_vaccine} (*{vaccine_info['desc']}*)\n\n"
        f"👉 **해당 바이러스에 대한 백신의 실제 감염 예방 효능**: **{int(actual_efficacy * 100)}%**")

# 5. 모델 연산 실행
gamma = virus_info["gamma"]
R0_val = virus_info["R0"]
beta = R0_val * gamma

t_arr, S, V, I, R = solve_svir_fast(
    N=N_pop, 
    beta=beta, 
    gamma=gamma, 
    v=daily_vac_rate_pct, 
    efficacy=actual_efficacy, 
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

# 6. 시각화 그래프
st.subheader("📊 시뮬레이션 결과 그래프 (단위: 백만 명)")

chart_data = {
    "S (감염 가능 미접종자)": S / 1e6,
    "V (백신 접종 완료자)": V / 1e6,
    "I (현재 감염자)": I / 1e6,
    "R (회복자/면역)": R / 1e6
}

st.line_chart(chart_data, height=400)
