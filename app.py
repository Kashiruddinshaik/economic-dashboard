import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import datetime

# App configuration
st.set_page_config(page_title="Economic Dashboard", layout="wide")

# Custom CSS styles
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .stApp { font-family: 'Segoe UI', sans-serif; }
        .block-container { padding: 2rem 2rem; }
        .title-text { font-size: 2.2rem; font-weight: bold; margin-bottom: 0.5rem; }
        .footer { font-size: 0.8rem; margin-top: 3rem; text-align: center; color: gray; }
        .metric-box { background-color: #1e1e1e; border-radius: 12px; padding: 1rem; text-align: center; margin-bottom: 1rem; }
        .insight-box { background-color: #262730; border-left: 6px solid #4CAF50; border-radius: 8px; padding: 1rem; margin-top: 1rem; font-size: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Load and clean data
try:
    df = pd.read_csv("data/clean_economic_data.csv")
    df = df[df["Country"].notnull()]
    df["Year"] = df["Year"].astype(str).str.replace(",", "").astype(int)
    df = df[df["Year"] >= 2000]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar controls
st.sidebar.title("Economic Dashboard")
st.sidebar.markdown("Analyze key economic indicators by country from 2000 onwards.")

countries = sorted(df["Country"].unique())
indicators = df.columns[2:]
selected_indicator = st.sidebar.selectbox("Select an indicator", indicators)

# Tab layout
tabs = st.tabs(["Single Country View", "Compare Countries"])

# Single Country Tab
with tabs[0]:
    selected_country = st.sidebar.selectbox("Select a country", countries)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"<div class='title-text'>{selected_country} – {selected_indicator} Over Time</div>", unsafe_allow_html=True)
        filtered = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()
        st.line_chart(filtered.set_index("Year"))

    with col2:
        latest_year = filtered["Year"].max()
        latest_value = filtered[filtered["Year"] == latest_year][selected_indicator].values[0]
        st.markdown(f"### Latest Value ({latest_year})")
        st.markdown(f"<div class='metric-box'><h2>{latest_value:,.2f}</h2></div>", unsafe_allow_html=True)

        first_year = filtered["Year"].min()
        first_value = filtered[filtered["Year"] == first_year][selected_indicator].values[0]
        growth = ((latest_value - first_value) / first_value) * 100
        insight = f"Between {first_year} and {latest_year}, <b>{selected_country}</b>'s <b>{selected_indicator}</b> changed by <b>{growth:.2f}%</b>."
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

    with st.expander("View Raw Data"):
        st.dataframe(filtered)
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered CSV", csv, file_name=f"{selected_country}_{selected_indicator}.csv")

    # GDP Forecast Section
    if selected_indicator == "GDP (current US$)":
        st.markdown("---")
        st.subheader(f"GDP Forecast for {selected_country} (Next 4 Years)")

        gdp_df = df[df["Country"] == selected_country][["Year", "GDP (current US$)"]].dropna()
        gdp_df = gdp_df.rename(columns={"Year": "ds", "GDP (current US$)": "y"})
        gdp_df["ds"] = pd.to_datetime(gdp_df["ds"], format="%Y")

        model = Prophet()
        model.fit(gdp_df)
        future = model.make_future_dataframe(periods=4, freq="Y")
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

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
        st.download_button("Download Forecast CSV", forecast_csv, file_name=f"{selected_country}_GDP_forecast.csv")

# Compare Countries Tab
with tabs[1]:
    multi_countries = st.multiselect("Select countries to compare", countries, default=["United States", "India"])
    if multi_countries:
        compare_df = df[df["Country"].isin(multi_countries)][["Country", "Year", selected_indicator]].dropna()
        pivot_df = compare_df.pivot(index="Year", columns="Country", values=selected_indicator)
        st.line_chart(pivot_df)
        st.dataframe(pivot_df)

# Footer
st.markdown(f"""
    <div class='footer'>
        Built by <a href='https://github.com/kashiruddinshaik' target='_blank'>Kashiruddin Shaik</a> | Powered by Streamlit + World Bank<br>
        Last updated: {datetime.today().strftime('%b %d, %Y')}
    </div>
""", unsafe_allow_html=True)
