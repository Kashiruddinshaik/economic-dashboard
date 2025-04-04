import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import openai

st.set_page_config(page_title="Economic Dashboard", layout="wide")

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
    <style>
        .main { background-color: #f7f7f7; }
        .stApp { font-family: 'Segoe UI', sans-serif; }
        .metric-card {
            background-color: #1c1c1c;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
        }
        .summary-box {
            background-color: #222;
            padding: 1rem;
            border-radius: 8px;
            border-left: 6px solid #4CAF50;
            font-size: 1rem;
            color: white;
        }
        .footer {
            font-size: 0.9rem;
            margin-top: 3rem;
            text-align: center;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

def load_data():
    df = pd.read_csv("data/clean_economic_data.csv")
    df = df[df["Country"].notnull()]
    df["Year"] = df["Year"].astype(str).str.replace(",", "").astype(int)
    return df[df["Year"] >= 2000]

df = load_data()
countries = sorted(df["Country"].unique())
indicators = df.columns[2:]

st.sidebar.title("🌐 Economic Dashboard")
st.sidebar.caption("Analyze key economic indicators by country from 2000 onwards.")
selected_indicator = st.sidebar.selectbox("Select an indicator", indicators)
selected_country = st.sidebar.selectbox("Select a country", countries)

filtered = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()
first_year = filtered["Year"].min()
latest_year = filtered["Year"].max()
first_value = filtered[filtered["Year"] == first_year][selected_indicator].values[0]
latest_value = filtered[filtered["Year"] == latest_year][selected_indicator].values[0]
growth = ((latest_value - first_value) / first_value) * 100

overview_tab, ai_tab, forecast_tab, compare_tab = st.tabs(["📈 Overview", "🤖 AI Insight", "🔮 Forecast", "🌍 Country Comparison"])

with overview_tab:
    st.markdown(f"## {selected_country} – {selected_indicator} Over Time")
    st.line_chart(filtered.set_index("Year"))
    st.markdown(f"<div class='metric-card'><h4>Latest Value ({latest_year})</h4><h2>{latest_value:,.2f}</h2></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='summary-box'>Between {first_year} and {latest_year}, <b>{selected_country}'s {selected_indicator}</b> changed by <b>{growth:.2f}%</b>.</div>", unsafe_allow_html=True)
    with st.expander("🔎 View Raw Data"):
        st.dataframe(filtered)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered CSV", csv, file_name=f"{selected_country}_{selected_indicator}.csv")

with ai_tab:
    st.markdown("## 🧠 AI-generated Insight")
    if st.button("Generate AI Insight Summary"):
        try:
            prompt = f"Summarize the trend of {selected_indicator} for {selected_country} from {first_year} to {latest_year}. Data: {filtered.to_dict(orient='records')}"
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

with forecast_tab:
    if selected_indicator == "GDP (current US$)":
        st.markdown(f"## 📈 GDP Forecast for {selected_country} (Next 4 Years)")
        gdp_df = df[df["Country"] == selected_country][["Year", "GDP (current US$)"]].dropna()
        gdp_df = gdp_df.rename(columns={"Year": "ds", "GDP (current US$)": "y"})
        gdp_df["ds"] = pd.to_datetime(gdp_df["ds"], format="%Y")

        model = Prophet()
        model.fit(gdp_df)
        future = model.make_future_dataframe(periods=4, freq="Y")
        forecast = model.predict(future)

        fig, ax = plt.subplots(figsize=(10, 5))
        model.plot(forecast, ax=ax)
        ax.set_title(f"{selected_country} GDP Forecast (Next 4 Years)", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("GDP (US$)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(4)
        forecast_df = forecast_df.rename(columns={
            "ds": "Year",
            "yhat": "Predicted GDP",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound"
        })
        forecast_df["Year"] = forecast_df["Year"].dt.year
        st.dataframe(forecast_df)
        forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", forecast_csv, file_name=f"{selected_country}_GDP_forecast.csv")
    else:
        st.info("GDP Forecasting is only available when 'GDP (current US$)' is selected.")

with compare_tab:
    st.markdown("## 🌍 Country Comparison")
    multi_countries = st.multiselect("Select countries to compare", countries)
    if multi_countries:
        compare_df = df[df["Country"].isin(multi_countries)][["Country", "Year", selected_indicator]].dropna()
        pivot_df = compare_df.pivot(index="Year", columns="Country", values=selected_indicator)
        st.line_chart(pivot_df)
        st.dataframe(pivot_df)

st.markdown(f"""
    <div class='footer'>
        Built by <a href='https://github.com/kashiruddinshaik' target='_blank'>Kashiruddin Shaik</a> — Powered by Streamlit & World Bank<br>
        Last updated: {datetime.today().strftime('%b %d, %Y')}
    </div>
""", unsafe_allow_html=True)
