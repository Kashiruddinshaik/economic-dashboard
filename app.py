import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import wbdata
from datetime import datetime
import openai
import os

st.set_page_config(page_title="Economic Dashboard", layout="wide")

# Load OpenAI API Key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Fetch data from World Bank
# Fetch data from World Bank
@st.cache_data(ttl=86400)
def load_data():
    indicators = {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "FP.CPI.TOTL.ZG": "Inflation (CPI %)",
        "SL.UEM.TOTL.ZS": "Unemployment (%)",
        "NE.EXP.GNFS.ZS": "Exports (% of GDP)"
    }
    # Get country ISO codes properly
    country_codes = [country.iso2c for country in wbdata.get_country()]
    
    df = wbdata.get_dataframe(
        indicators,
        country=country_codes,
        data_date=(datetime(2000, 1, 1), datetime.today()),
        convert_date=True
    )
    df.reset_index(inplace=True)
    df.rename(columns={"country": "Country", "date": "Year"}, inplace=True)
    df["Year"] = pd.to_datetime(df["Year"]).dt.year
    df = df[df["Year"] >= 2000]
    return df


# Generate AI-based insight using OpenAI
@st.cache_data(show_spinner=False)
def generate_ai_insight(text):
    prompt = f"Generate a summary for this economic indicator trend:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Load data
df = load_data()

# Sidebar
st.sidebar.markdown("## 🌍 Economic Dashboard")
st.sidebar.markdown("Analyze key economic indicators by country from 2000 onwards.")

indicators = df.columns[2:]
selected_indicator = st.sidebar.selectbox("Select an indicator", indicators)
selected_country = st.sidebar.selectbox("Select a country", sorted(df["Country"].unique()))

# Filter data
country_df = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()

# Main Panel
st.markdown(f"### {selected_country} - {selected_indicator} Over Time")
st.line_chart(data=country_df, x="Year", y=selected_indicator)

latest_year = country_df["Year"].max()
latest_value = country_df[country_df["Year"] == latest_year][selected_indicator].values[0]
initial_year = country_df["Year"].min()
initial_value = country_df[country_df["Year"] == initial_year][selected_indicator].values[0]
percentage_change = ((latest_value - initial_value) / initial_value) * 100

st.metric(
    label=f"Latest Value ({latest_year})",
    value=f"{latest_value:,.2f}",
    delta=f"{percentage_change:.2f}% change since {initial_year}"
)

# AI Summary
if st.button("Generate AI Insight Summary"):
    insight = generate_ai_insight(country_df.to_csv(index=False))
    st.success(insight)

# Raw Data
with st.expander("📊 View Raw Data"):
    st.dataframe(country_df)
    st.download_button("Download Filtered CSV", data=country_df.to_csv(index=False), file_name=f"{selected_country}_{selected_indicator}.csv")

# Footer
st.markdown("""
---
Built by [Kashiruddin Shaik](https://github.com/Kashiruddinshaik) — Powered by Streamlit & World Bank  
Last updated: Apr 04, 2025
""")
