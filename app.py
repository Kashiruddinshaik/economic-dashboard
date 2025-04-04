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
@st.cache_data(ttl=86400)
def load_data():
    indicators = {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "FP.CPI.TOTL.ZG": "Inflation (CPI %)",
        "SL.UEM.TOTL.ZS": "Unemployment (%)",
        "NE.EXP.GNFS.ZS": "Exports (% of GDP)"
    }
    try:
        # Get country ISO codes
        countries = wbdata.get_country()
        if not countries:
            st.warning("No countries retrieved from wbdata.get_country(). Using fallback country codes.")
            # Fallback list of country codes
            country_codes = ['US', 'CN', 'IN', 'GB', 'FR', 'DE', 'JP', 'BR', 'AU', 'CA']
        else:
            # Debug: Print a sample of the data to inspect
            st.write("Sample countries:", countries[:5])
            # Extract ISO2 codes
            country_codes = []
            for country in countries:
                if 'iso2Code' in country:
                    country_codes.append(country['iso2Code'])
                else:
                    st.write(f"Missing 'iso2Code' in country: {country}")
            if not country_codes:
                st.warning("No valid country codes found. Using fallback country codes.")
                country_codes = ['US', 'CN', 'IN', 'GB', 'FR', 'DE', 'JP', 'BR', 'AU', 'CA']

        # Fetch data
        df = wbdata.get_dataframe(
            indicators,
            country=country_codes,
            data_date=(datetime(2000, 1, 1), datetime.today()),
            convert_date=True
        )
        if df.empty:
            raise ValueError("No data retrieved from wbdata.get_dataframe()")
        
        df.reset_index(inplace=True)
        df.rename(columns={"country": "Country", "date": "Year"}, inplace=True)
        df["Year"] = pd.to_datetime(df["Year"]).dt.year
        df = df[df["Year"] >= 2000]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe to prevent app crash

# Generate AI-based insight using OpenAI
@st.cache_data(show_spinner=False)
def generate_ai_insight(text):
    try:
        prompt = f"Generate a summary for this economic indicator trend:\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating insight: {str(e)}"

# Load data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("Failed to load data. Please check the logs or try again later.")
    st.stop()

# Sidebar
st.sidebar.markdown("## 🌍 Economic Dashboard")
st.sidebar.markdown("Analyze key economic indicators by country from 2000 onwards.")

indicators = df.columns[2:]
selected_indicator = st.sidebar.selectbox("Select an indicator", indicators)
selected_country = st.sidebar.selectbox("Select a country", sorted(df["Country"].unique()))

# Filter data
country_df = df[df["Country"] == selected_country][["Year", selected_indicator]].dropna()

# Main Panel
if country_df.empty:
    st.warning(f"No data available for {selected_country} and {selected_indicator}.")
else:
    st.markdown(f"### {selected_country} - {selected_indicator} Over Time")
    st.line_chart(data=country_df, x="Year", y=selected_indicator)

    latest_year = country_df["Year"].max()
    latest_value = country_df[country_df["Year"] == latest_year][selected_indicator].values[0]
    initial_year = country_df["Year"].min()
    initial_value = country_df[country_df["Year"] == initial_year][selected_indicator].values[0]
    if initial_value == 0:
        percentage_change = 0
    else:
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
