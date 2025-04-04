import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime
import openai

st.set_page_config(page_title="Economic Dashboard", layout="wide")

st.markdown("""
    <style>
        body, .main, .stApp {
            background-color: #111;
            color: #f0f0f0;
            font-family: 'Segoe UI', sans-serif;
        }
        .title-text {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .metric-box {
            background-color: #222;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1rem;
        }
        .insight-box {
            background-color: #1b1e23;
            border-left: 5px solid #2ecc71;
            padding: 1rem;
            margin-top: 1rem;
            border-radius: 5px;
        }
        .footer {
            font-size: 0.9rem;
            text-align: center;
            color: #888;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

try:
    df = pd.read_csv("data/clean_economic_data.csv")
    df = df[df["Country"].notnull()]
    df["Year"] = df["Year"].astype(str).str.replace(",", "").astype(int)
    df = df[df["Year"] >= 2000]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.title("🌍 Economic Dashboard")
st.sidebar.write("Analyze key economic indicators by country from 2000 onwards.")
countries = sorted(df["Country"].dropna().unique())
indicators = df.columns[2:]
selected_indicator = st.sidebar.selectbox("Select an indicator", indicators)
selected_country = st.sidebar.selectbox("Select a country", countries)

filtered = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()

st.markdown(f"<div class='title-text'>{selected_country} – {selected_indicator} Over Time</div>", unsafe_allow_html=True)
st.line_chart(filtered.set_index("Year"))

latest_year = filtered["Year"].max()
latest_value = filtered[filtered["Year"] == latest_year][selected_indicator].values[0]
st.markdown(f"<div class='metric-box'><h4>Latest Value ({latest_year})</h4><h2>{latest_value:,.2f}</h2></div>", unsafe_allow_html=True)

first_year = filtered["Year"].min()
first_value = filtered[filtered["Year"] == first_year][selected_indicator].values[0]
growth = ((latest_value - first_value) / first_value) * 100
insight = f"Between {first_year} and {latest_year}, <b>{selected_country}</b>'s <b>{selected_indicator}</b> changed by <b>{growth:.2f}%</b>."
st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

if st.button("Generate AI Insight Summary"):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        prompt = f"Summarize the trend of {selected_indicator} for {selected_country} from {first_year} to {latest_year}. Data: {filtered.to_dict(orient='records')}"
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        summary = response.choices[0].message.content
        st.markdown(f"<div class='insight-box'>{summary}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating summary: {e}")

with st.expander("🔎 View Raw Data"):
    st.dataframe(filtered)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered CSV", csv, file_name=f"{selected_country}_{selected_indicator}.csv")

if selected_indicator == "GDP (current US$)":
    st.markdown("---")
    st.subheader(f"📈 GDP Forecast for {selected_country} (Next 4 Years)")

    gdp_df = df[df["Country"] == selected_country][["Year", "GDP (current US$)"]].dropna()
    gdp_df = gdp_df.rename(columns={"Year": "ds", "GDP (current US$)": "y"})
    gdp_df["ds"] = pd.to_datetime(gdp_df["ds"], format="%Y")

    model = Prophet()
    model.fit(gdp_df)
    future = model.make_future_dataframe(periods=4, freq="Y")
    forecast = model.predict(future)

    st.pyplot(model.plot(forecast))

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

st.markdown("---")
st.subheader(f"🌐 Compare {selected_indicator} Across Countries")
multi_countries = st.multiselect("Select countries to compare", countries)
if multi_countries:
    compare_df = df[df["Country"].isin(multi_countries)][["Country", "Year", selected_indicator]].dropna()
    pivot_df = compare_df.pivot(index="Year", columns="Country", values=selected_indicator)
    st.line_chart(pivot_df)
    st.dataframe(pivot_df)

st.markdown(f"""
    <div class='footer'>
        Built by <a href='https://github.com/kashiruddinshaik' target='_blank'>Kashiruddin Shaik</a> |
        Powered by Streamlit + World Bank<br>
        Last updated: {datetime.today().strftime('%b %d, %Y')}
    </div>
""", unsafe_allow_html=True)
