import streamlit as st
import numpy as np
import pandas as pd

# 웹페이지 기본 설정
st.set_page_config(
    page_title="대한민국 SVIR 바이러스-백신 시뮬레이터",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 대한민국 SVIR 감염병 - 백신 시뮬레이터")
st.markdown("""
이 시뮬레이터는 **SVIR 모델(Susceptible-Vaccinated-Infected-Recovered)**을 사용하여 바이러스 변이 및 백신 종류에 따른 감염 확산 양상을 시각화합니다.
""")

# 1. 바이러스 정의 (간염 제거)
VIRUSES = {
    "COVID-19 (원형)": {"R0": 2.5, "gamma": 1/14, "desc": "초기 코로나19 바이러스 (R0 = 2.5)"},
    "COVID-19 (델타 변이)": {"R0": 5.0, "gamma": 1/10, "desc": "높은 전파력의 델타 변이 (R0 = 5.0)"},
    "COVID-19 (오미크론 변이)": {"R0": 10.0, "gamma": 1/7, "desc": "극도로 빠른 전파력의 오미크론 변이 (R0 = 10.0)"},
    "SARS": {"R0": 3.0, "gamma": 1/12, "desc": "사스 중증급성호흡기증후군 (R0 = 3.0)"}
}

# 2. 백신 효능 매트릭스 (간염 항목 제거)
VACCINE_EFFICACY_MATRIX = {
    "mRNA 백신": {
        "COVID-19 (원형)": 0.95,
        "COVID-19 (델타 변이)": 0.85,
        "COVID-19 (오미크론 변이)": 0.60,
        "SARS": 0.70,
        "desc": "변이 바이러스 감염 예방 우수"
    },
    "생백신 (약독화)": {
        "COVID-19 (원형)": 0.70,
        "COVID-19 (델타 변이)": 0.50,
        "COVID-19 (오미크론 변이)": 0.30,
        "SARS": 0.60,
        "desc": "전통적 약독화 백신 (중등도 효능)"
    },
    "사백신 (불활성화)": {
        "COVID-19 (원형)": 0.50,
        "COVID-19 (델타 변이)": 0.30,
        "COVID-19 (오미크론 변이)": 0.10,
        "SARS": 0.40,
        "desc": "불활성화 백신 (변이 바이러스 예방 효과 저하)"
    }
}

# 3. 수치해석 함수
def solve_svir(N, beta, gamma, v, efficacy, I0, initial_vac_pct, days):
    dt = 1.0
    steps = int(days)
    
    t_arr = np.linspace(0, days, steps)
    S = np.zeros(steps)
    V = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    
    V[0] = (N - I0) * initial_vac_pct
    S[0] = (N - I0) - V[0]
    I[0] = I0
    R[0] = 0.0
    
    beta_v = beta * (1.0 - efficacy)
    
    for i in range(steps - 1):
        s_curr, v_curr, i_curr = S[i], V[i], I[i]
        
        dS = (-beta * s_curr * i_curr / N - v * s_curr) * dt
        dV = (v * s_curr - beta_v * v_curr * i_curr / N) * dt
        dI = (beta * s_curr * i_curr / N + beta_v * v_curr * i_curr / N - gamma * i_curr) * dt
        dR = (gamma * i_curr) * dt
        
        S[i+1] = max(0.0, s_curr + dS)
        V[i+1] = max(0.0, v_curr + dV)
        I[i+1] = max(0.0, i_curr + dI)
        R[i+1] = max(0.0, R[i] + dR)
        
    return t_arr, S, V, I, R

# 4. 사이드바 조작창
st.sidebar.header("⚙️ 시뮬레이션 파라미터 설정")

selected_virus = st.sidebar.selectbox("1. 바이러스 종류 선택", list(VIRUSES.keys()), index=0)
selected_vaccine = st.sidebar.selectbox("2. 백신 종류 선택", list(VACCINE_EFFICACY_MATRIX.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 시뮬레이션 환경 조건")
N_pop = st.sidebar.number_input("대한민국 인구수 (명)", value=51600000, step=1000000, format="%d")
initial_infected = st.sidebar.number_input("초기 감염자 수 (명)", value=100, step=10)
initial_vac_pct = st.sidebar.slider("시작 시점 접종 완료율 (%)", min_value=0, max_value=90, value=50, step=5) / 100.0
daily_vac_rate_pct = st.sidebar.slider("일일 추가 백신 접종률 (%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1) / 100.0
sim_days = st.sidebar.slider("시뮬레이션 기간 (일)", min_value=30, max_value=365, value=180, step=10)

virus_info = VIRUSES[selected_virus]
vaccine_info = VACCINE_EFFICACY_MATRIX[selected_vaccine]
actual_efficacy = vaccine_info[selected_virus]

st.info(
    f"**선택된 바이러스**: {selected_virus} (*{virus_info['desc']}*)\n\n"
    f"**선택된 백신**: {selected_vaccine} (*{vaccine_info['desc']}*)\n\n"
    f"👉 **적용된 실제 감염 예방 효능(Efficacy)**: **{int(actual_efficacy * 100)}%**"
)

gamma = virus_info["gamma"]
R0_val = virus_info["R0"]
beta = R0_val * gamma

# 5. 선택 백신 시뮬레이션
t_arr, S, V, I, R = solve_svir(
    N=N_pop, beta=beta, gamma=gamma, v=daily_vac_rate_pct, 
    efficacy=actual_efficacy, I0=initial_infected, 
    initial_vac_pct=initial_vac_pct, days=sim_days
)

max_infected = int(np.max(I))
peak_idx = np.argmax(I)
peak_day = int(t_arr[peak_idx])

col1, col2, col3, col4 = st.columns(4)
col1.metric("최대 동시 감염자", f"{max_infected:,} 명")
col2.metric("감염 피크(정점)", f"{peak_day} 일째")
col3.metric("최종 백신 접종자", f"{int(V[-1]):,} 명")
col4.metric("최종 회복/면역자", f"{int(R[-1]):,} 명")

st.markdown("---")

# 6. 탭 구성
tab1, tab2 = st.tabs(["📊 현재 백신 시뮬레이션 결과", "⚔️ 3가지 백신 감염자 수($I$) 직접 비교"])

with tab1:
    st.subheader("1. 전체 군집 변화 그래프 (단위: 백만 명)")
    df_all = pd.DataFrame({
        "날짜 (일)": t_arr,
        "S (감염 가능 미접종자)": S / 1e6,
        "V (백신 접종 완료자)": V / 1e6,
        "I (현재 감염자)": I / 1e6,
        "R (회복자/면역)": R / 1e6
    }).set_index("날짜 (일)")
    st.line_chart(df_all, height=320)

    st.subheader("2. 🔥 [확대] 현재 감염자 수($I$) 추이 (단위: 명)")
    st.caption("전체 인구수에 묻히지 않도록 감염자 수만 별도로 확대한 그래프입니다.")
    df_inf_only = pd.DataFrame({
        "날짜 (일)": t_arr,
        "현재 감염자 수 (명)": I
    }).set_index("날짜 (일)")
    st.line_chart(df_inf_only, height=300)

with tab2:
    st.subheader(f"🦠 [{selected_virus}] 바이러스에 대한 백신 종류별 감염자 수 비교")
    st.write("동일한 조건에서 백신 종류에 따라 감염자 수($I$) 확산 추이가 어떻게 달라지는지 한눈에 비교합니다.")
    
    comparison_data = {"날짜 (일)": t_arr}
    
    for vac_name, vac_dict in VACCINE_EFFICACY_MATRIX.items():
        eff = vac_dict[selected_virus]
        _, _, _, I_comp, _ = solve_svir(
            N=N_pop, beta=beta, gamma=gamma, v=daily_vac_rate_pct,
            efficacy=eff, I0=initial_infected,
            initial_vac_pct=initial_vac_pct, days=sim_days
        )
        comparison_data[f"{vac_name} (효능 {int(eff*100)}%)"] = I_comp
        
    df_comp = pd.DataFrame(comparison_data).set_index("날짜 (일)")
    st.line_chart(df_comp, height=420)
