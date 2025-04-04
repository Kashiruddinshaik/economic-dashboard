# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import io

# Streamlit page config
st.set_page_config(layout="wide")

# Load data with error handling and preview
try:
    df = pd.read_csv("data/clean_economic_data.csv")

    # Preview top rows to debug
    st.write("📊 Preview of uploaded dataset:")
    st.dataframe(df.head(10))

    # Clean and filter dataset
    df = df[df["Country"].notnull()]
    df["Year"] = df["Year"].astype(str).str.replace(",", "").astype(int)
    df = df[df["Year"] >= 2000]

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Sidebar controls
st.sidebar.title("🌍 Economic Dashboard")
countries = df["Country"].dropna().unique()
if len(countries) == 0:
    st.error("No data available. Please check your dataset.")
    st.stop()

selected_country = st.sidebar.selectbox("Select a country", countries)
selected_indicator = st.sidebar.selectbox("Select an indicator", df.columns[2:])

# Main plot section
filtered = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()

st.title(f"{selected_country} - {selected_indicator} Over Time")
st.line_chart(filtered.set_index("Year"))

# Expand raw data
with st.expander("📄 Show raw data"):
    st.dataframe(filtered)

# Download button for filtered data
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"{selected_country}_{selected_indicator}_data.csv",
    mime='text/csv'
)

# Forecast section for GDP
if selected_indicator == "GDP (current US$)":
    st.subheader(f"🔮 GDP Forecast for {selected_country} (2024–2027)")

    # Prepare GDP data for Prophet
    gdp_df = df[df["Country"] == selected_country][["Year", "GDP (current US$)"]].dropna()
    gdp_df = gdp_df.rename(columns={"Year": "ds", "GDP (current US$)": "y"})
    gdp_df["ds"] = pd.to_datetime(gdp_df["ds"], format="%Y")

    # Build and train Prophet model
    model = Prophet()
    model.fit(gdp_df)
    future = model.make_future_dataframe(periods=4, freq='Y')
    forecast = model.predict(future)

    # Plot forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Show forecasted values
    forecast_output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(4).rename(columns={
        "ds": "Year",
        "yhat": "Predicted GDP",
        "yhat_lower": "Lower Bound",
        "yhat_upper": "Upper Bound"
    })

    st.write("### Forecasted GDP:")
    st.dataframe(forecast_output)

    # Download button for forecast
    forecast_csv = forecast_output.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=forecast_csv,
        file_name=f"{selected_country}_GDP_forecast.csv",
        mime='text/csv'
    )

# Footer
st.caption("Built with Streamlit | Data Source: World Bank")
