import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv
import os

# ==========================
# Load API Key
# ==========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY tidak ditemukan di .env atau st.secrets. Harap isi terlebih dahulu.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==========================
# Streamlit Config
# ==========================
st.set_page_config(
    page_title="Monthly Report Copilot AI – Scenario Planning",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Monthly Report Copilot AI – Scenario Planning & Strategic Insights 🤖")

# ==========================
# Sidebar Model Selection
# ==========================
model = st.sidebar.selectbox(
    "Pilih Model AI",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
)

# ==========================
# File Upload
# ==========================
uploaded_file = st.file_uploader("📥 Upload File Excel (wajib ada kolom Description & Amount)", type=["xlsx"])

df = None
scenario_df = None

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        st.stop()

    # Validasi kolom
    required_cols = {"Description", "Amount"}
    if not required_cols.issubset(df.columns):
        st.error("❌ File wajib memiliki kolom: Description dan Amount")
        st.stop()

    # ==========================
    # Buat Scenario Planning
    # ==========================
    np.random.seed(42)
    optimistic_factor = np.random.uniform(1.05, 1.25)
    pessimistic_factor = np.random.uniform(0.85, 0.95)
    worst_factor = np.random.uniform(0.60, 0.80)

    scenario_df = df.copy()
    scenario_df["Optimistic"] = scenario_df["Amount"] * optimistic_factor
    scenario_df["Pessimistic"] = scenario_df["Amount"] * pessimistic_factor
    scenario_df["Worst Case"] = scenario_df["Amount"] * worst_factor

    # ==========================
    # UI Layout 2 Kolom
    # ==========================
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("📁 Data & Scenario Output")
        st.dataframe(scenario_df)

        # ==========================
        # Plotly Chart
        # ==========================
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=scenario_df["Description"],
            y=scenario_df["Optimistic"],
            name="Optimistic"
        ))
        fig.add_trace(go.Bar(
            x=scenario_df["Description"],
            y=scenario_df["Pessimistic"],
            name="Pessimistic"
        ))
        fig.add_trace(go.Bar(
            x=scenario_df["Description"],
            y=scenario_df["Worst Case"],
            name="Worst Case"
        ))

        fig.update_layout(
            barmode='group',
            title="📊 Scenario Projection Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

# ==========================
# AI Analysis Function
# ==========================
def generate_analysis(df_preview):
    system_prompt = """
Kamu adalah Financial Copilot AI. Analisislah EBIT, revenue, margin, OPEX, CAPEX,
COGS, cash flow, likuiditas, firm value, ESG, OPEX to Revenue, Gross Profit Margin,
Operating Profit Margin, Net Profit Margin, Degree of Operating Leverage, financial distress,
risiko, dan strategi pertumbuhan. Berikan rekomendasi jangka pendek dan jangka panjang.
"""

    user_prompt = f"""
Berikut data scenario (maks 20 baris):
{df_preview.to_string(index=False)}

Berikan analisis finansial lengkap.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return response.choices[0].message.content   # ✔ FIXED


# ==========================
# Chatbot Initialize
# ==========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================
# Right Panel (AI Analysis + Chatbot)
# ==========================
with st.sidebar:
    st.markdown("## 💬 Copilot Chat")

with st.container():
    if scenario_df is not None:
        with right:
            st.subheader("🤖 AI Financial Analysis")

            df_preview = scenario_df.head(20)
            analysis = generate_analysis(df_preview)
            st.write(analysis)

            st.subheader("💬 Chatbot")

            # Display chat history
            for sender, msg in st.session_state.chat_history:
                if sender == "user":
                    st.markdown(f"**👤 You:** {msg}")
                else:
                    st.markdown(f"**🤖 Copilot:** {msg}")

            # Chat input
            user_input = st.text_input("Tanya sesuatu tentang finansial...", key="chat_input")
            send = st.button("Kirim")
            reset = st.button("Reset Chat")

            if send and user_input:
                # Append user message
                st.session_state.chat_history.append(("user", user_input))

                # AI reply
                chat_response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Kamu adalah Financial Copilot AI."},
                        *[
                            {"role": "user" if s == "user" else "assistant", "content": m}
                            for s, m in st.session_state.chat_history
                        ],
                    ]
                )

                ai_msg = chat_response.choices[0].message.content   # ✔ FIXED

                st.session_state.chat_history.append(("assistant", ai_msg))
                st.rerun()

            if reset:
                st.session_state.chat_history = []
                st.success("🔄 Chat berhasil direset!")
                st.rerun()
