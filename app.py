import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import openai
import pycountry

st.set_page_config(page_title="Global Economic Dashboard", layout="wide")

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
    <style>
        .stApp { font-family: 'Segoe UI', sans-serif; }
        .metric-card {
            background-color: #1c1c1c;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        }
        .summary-box {
            background-color: #222;
            padding: 1rem;
            border-radius: 8px;
            border-left: 6px solid #4CAF50;
            font-size: 1rem;
            color: white;
            margin-top: 1rem;
        }
        .footer {
            font-size: 0.9rem;
            margin-top: 3rem;
            text-align: center;
            color: #888;
        }
        .title-section {
            text-align: center;
            margin-top: 1rem;
        }
        .title-section h1 {
            font-size: 2.5rem;
            color: #00BFFF;
        }
        .title-section p {
            font-size: 1.1rem;
            color: #bbb;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-section">
        <h1>Global Economic Dashboard</h1>
        <p>Visualize and analyze key economic indicators across countries</p>
    </div>
""", unsafe_allow_html=True)

def get_flag(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        return chr(127397 + ord(country.alpha_2[0])) + chr(127397 + ord(country.alpha_2[1]))
    except:
        return ""

def load_data():
    df = pd.read_csv("data/clean_economic_data.csv")
    df = df[df["Country"].notnull()]
    df["Year"] = df["Year"].astype(str).str.replace(",", "").astype(int)
    return df[df["Year"] >= 2000]

df = load_data()
countries = sorted(df["Country"].unique())
indicators = df.columns[2:]

st.sidebar.title("Navigation")
st.sidebar.caption("Choose country and indicator")
selected_indicator = st.sidebar.selectbox("Select an indicator", indicators)
selected_country = st.sidebar.selectbox("Select a country", countries)

filtered = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()
first_year = filtered["Year"].min()
latest_year = filtered["Year"].max()
first_value = filtered[filtered["Year"] == first_year][selected_indicator].values[0]
latest_value = filtered[filtered["Year"] == latest_year][selected_indicator].values[0]
growth = ((latest_value - first_value) / first_value) * 100

flag = get_flag(selected_country)
header_title = f"{flag} {selected_country} — {selected_indicator}"

overview_tab, ai_tab, compare_tab = st.tabs(["Overview", "AI Insight", "Country Comparison"])

with overview_tab:
    st.markdown(f"## {header_title}")
    st.caption("Trend over time for the selected indicator")
    st.line_chart(filtered.set_index("Year"))
    col1, col2 = st.columns(2)
    col1.markdown(f"<div class='metric-card'><h4>Latest Value ({latest_year})</h4><h2>{latest_value:,.2f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h4>Growth Since {first_year}</h4><h2>{growth:.2f}%</h2></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary-box'>Between {first_year} and {latest_year}, <b>{selected_country}'s {selected_indicator}</b> changed by <b>{growth:.2f}%</b>.</div>", unsafe_allow_html=True)
    with st.expander("View Raw Data"):
        st.dataframe(filtered)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered CSV", csv, file_name=f"{selected_country}_{selected_indicator}.csv")

with ai_tab:
    st.markdown("## AI-generated Insight")
    if st.button("Generate Summary with GPT"):
        try:
            prompt = f"Summarize the trend of {selected_indicator} for {selected_country} from {first_year} to {latest_year}. Data: {filtered.to_dict(orient='records')}. Keep it concise."
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=150
            )
            summary = response.choices[0].message.content
            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating summary: {e}")

with compare_tab:
    st.markdown("## Country Comparison")
    multi_countries = st.multiselect("Compare countries", countries)
    if multi_countries:
        compare_df = df[df["Country"].isin(multi_countries)][["Country", "Year", selected_indicator]].dropna()
        if compare_df.empty:
            st.warning("No data available for the selected indicator across the chosen countries.")
        else:
            pivot_df = compare_df.pivot(index="Year", columns="Country", values=selected_indicator)
            st.line_chart(pivot_df)
            st.dataframe(pivot_df)

st.markdown(f"""
    <div class='footer'>
        Built by <a href='https://github.com/kashiruddinshaik' target='_blank'>Kashiruddin Shaik</a> — Powered by Streamlit & World Bank<br>
        Last updated: {datetime.today().strftime('%b %d, %Y')}
    </div>
""", unsafe_allow_html=True)
