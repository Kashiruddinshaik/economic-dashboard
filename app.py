import streamlit as st
import pandas as pd
import wbdata
import datetime
from datetime import datetime
import plotly.express as px
from prophet import Prophet
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🌍 Global Economic Dashboard - Real-Time Data")


st.caption(f"⏱️ Last updated (Local Time): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Auto-refreshed every 24h")


# --- CONFIG --- #
INDICATORS = {
    "FP.CPI.TOTL.ZG": "Inflation (CPI)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SL.UEM.TOTL.ZS": "Unemployment Rate","FP.CPI.TOTL.ZG": "Inflation (CPI)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SL.UEM.TOTL.ZS": "Unemployment Rate",
    "SP.POP.TOTL": "Total Population",
    "NY.GNP.PCAP.CD": "GNI per Capita (US$)",
    "NE.EXP.GNFS.CD": "Exports of Goods & Services (US$)",
    "NE.IMP.GNFS.CD": "Imports of Goods & Services (US$)",
    "GC.TAX.TOTL.GD.ZS": "Tax Revenue (% of GDP)",
    "SE.XPD.TOTL.GD.ZS": "Gov. Expenditure on Education (% of GDP)",
    "SH.XPD.CHEX.GD.ZS": "Health Expenditure (% of GDP)"
}

COUNTRY_ISO_MAP = {
    "USA": "🇺🇸 United States", "CHN": "🇨🇳 China", "JPN": "🇯🇵 Japan", "DEU": "🇩🇪 Germany",
    "IND": "🇮🇳 India", "GBR": "🇬🇧 United Kingdom", "FRA": "🇫🇷 France", "BRA": "🇧🇷 Brazil",
    "ITA": "🇮🇹 Italy", "CAN": "🇨🇦 Canada", "KOR": "🇰🇷 South Korea", "RUS": "🇷🇺 Russia",
    "AUS": "🇦🇺 Australia", "ESP": "🇪🇸 Spain", "MEX": "🇲🇽 Mexico", "IDN": "🇮🇩 Indonesia",
    "NLD": "🇳🇱 Netherlands", "SAU": "🇸🇦 Saudi Arabia", "TUR": "🇹🇷 Turkey", "CHE": "🇨🇭 Switzerland",
    "ARG": "🇦🇷 Argentina", "SWE": "🇸🇪 Sweden", "POL": "🇵🇱 Poland", "BEL": "🇧🇪 Belgium",
    "THA": "🇹🇭 Thailand", "IRN": "🇮🇷 Iran", "AUT": "🇦🇹 Austria", "NOR": "🇳🇴 Norway",
    "ARE": "🇦🇪 UAE", "NGA": "🇳🇬 Nigeria", "ISR": "🇮🇱 Israel", "IRL": "🇮🇪 Ireland",
    "SGP": "🇸🇬 Singapore", "ZAF": "🇿🇦 South Africa", "PHL": "🇵🇭 Philippines",
    "EGY": "🇪🇬 Egypt", "COL": "🇨🇴 Colombia", "MYS": "🇲🇾 Malaysia", "PAK": "🇵🇰 Pakistan",
    "CHL": "🇨🇱 Chile", "FIN": "🇫🇮 Finland", "VNM": "🇻🇳 Vietnam", "CZE": "🇨🇿 Czech Republic",
    "ROU": "🇷🇴 Romania", "PRT": "🇵🇹 Portugal", "PER": "🇵🇪 Peru", "NZL": "🇳🇿 New Zealand",
    "UKR": "🇺🇦 Ukraine", "HUN": "🇭🇺 Hungary"
}

# Reverse lookup for display name to ISO code
COUNTRIES = {v: k for k, v in COUNTRY_ISO_MAP.items()}

# --- Sidebar Filters --- #
st.sidebar.header("🔍 Filter")

selected_indicator_name = st.sidebar.selectbox("Select an Indicator", list(INDICATORS.values()))
selected_indicator_code = [code for code, name in INDICATORS.items() if name == selected_indicator_name][0]

selected_countries = st.sidebar.multiselect("Select Countries", list(COUNTRIES.keys()), default=["🇺🇸 United States", "🇮🇳 India"])

year_range = st.sidebar.slider("Select Year Range", 2000, 2023, (2000, 2023))

# --- Fetch data --- #
@st.cache_data(ttl=86400, show_spinner=True)  # Auto-refresh daily
def fetch_data(selected_indicator_code):
    start_date = datetime(2000, 1, 1)

    end_date = datetime(2023, 1, 1)

    try:
        iso_codes = list(COUNTRY_ISO_MAP.keys())
        raw_df = wbdata.get_dataframe(
            {selected_indicator_code: selected_indicator_name},
            country=iso_codes,
            date=(start_date, end_date)
        ).reset_index()

        if 'country' in raw_df.columns and 'date' in raw_df.columns:
            raw_df.rename(columns={
                'country': 'Country',
                'date': 'Year',
                selected_indicator_name: 'Value'
            }, inplace=True)
            raw_df['Year'] = pd.to_datetime(raw_df['Year']).dt.year
            raw_df['Country'] = raw_df['Country'].map(
                lambda name: next((v for k, v in COUNTRY_ISO_MAP.items() if name.lower() in v.lower()), name)
            )
            return raw_df[['Year', 'Country', 'Value']]
        else:
            return pd.DataFrame(columns=['Year', 'Country', 'Value'])

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame(columns=['Year', 'Country', 'Value'])


# --- Load data --- #
with st.spinner("Fetching live economic data from World Bank..."):
    data = fetch_data(selected_indicator_code)

# --- Filter data --- #
data = data[(data['Year'] >= year_range[0]) & (data['Year'] <= year_range[1])]
data = data[data['Country'].isin(selected_countries)]

# --- Tabs --- #
tabs = st.tabs(["Overview", "Comparison", "Map", "Forecast", "Download"])

# --- Overview Tab --- #
with tabs[0]:
    st.subheader(f"📊 {selected_indicator_name} Trends")
    for country in selected_countries:
        country_data = data[data['Country'] == country].sort_values("Year")
        if not country_data.empty:
            st.metric(label=f"Latest {selected_indicator_name} - {country}",
                      value=f"{country_data['Value'].iloc[-1]:,.2f}",
                      delta=f"{(country_data['Value'].iloc[-1] - country_data['Value'].iloc[-2]):+.2f}" if len(country_data) > 1 else "")
            st.line_chart(country_data.set_index("Year")["Value"])
        else:
            st.info(f"No data available for {country}.")

# --- Comparison Tab --- #
with tabs[1]:
    st.subheader(f"🔍 Comparison - {selected_indicator_name}")
    if not data.empty:
        year_counts = data.groupby("Year")["Value"].count()
        recent_valid_year = year_counts[year_counts >= 2].index.max() if not year_counts.empty else None
        compare_df = data[data['Year'] == recent_valid_year] if recent_valid_year else pd.DataFrame()
        if not compare_df.empty:
            st.bar_chart(compare_df.set_index("Country")["Value"])
        else:
            st.info("No comparison data available for the selected range.")
    else:
        st.info("No comparison data available.")

# --- Map Tab --- #
with tabs[2]:
    st.subheader("🗺 Indicator Map View")
    try:
        reverse_map = {v: k for k, v in COUNTRY_ISO_MAP.items()}
        map_df = data.copy()
        map_df["iso_code"] = map_df["Country"].map(reverse_map)
        fig = px.choropleth(map_df, locations="iso_code", color="Value",
                            hover_name="Country", animation_frame="Year",
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title=f"{selected_indicator_name} by Country")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Map failed to render: {e}")

# --- Forecast Tab --- #
# --- Forecast Tab --- #
with tabs[3]:
    st.subheader(f"🔮 Forecasting - {selected_indicator_name}")
    forecast_horizon = st.slider("📅 Forecast years", 1, 15, 5)
    valid_forecasts = []

    for country in selected_countries:
        country_data = data[data['Country'] == country].dropna().sort_values("Year")
        if len(country_data) >= 5:
            df_prophet = country_data.rename(columns={"Year": "ds", "Value": "y"})
            df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=forecast_horizon, freq='Y')
            forecast = model.predict(future)

            # ✅ Correct mapping of actual values
            actual_map = df_prophet.set_index("ds")["y"].to_dict()
            forecast["Actual"] = forecast["ds"].map(actual_map)

            forecast["Country"] = country
            valid_forecasts.append(forecast[["ds", "yhat", "Country", "Actual"]].copy())

    if valid_forecasts:
        forecast_df = pd.concat(valid_forecasts)
        fig = px.line(forecast_df, x="ds", y="yhat", color="Country", title="Multi-Country Forecast")
        for country in selected_countries:
            actual = forecast_df[(forecast_df["Country"] == country) & forecast_df["Actual"].notna()]
            fig.add_scatter(x=actual["ds"], y=actual["Actual"], mode="markers", name=f"Actual - {country}")
        st.plotly_chart(fig, use_container_width=True)

        # --- AI-Generated Insights --- #
        st.subheader("🧠 AI Insights")
        insights_data = []
        for country in selected_countries:
            country_forecast = forecast_df[forecast_df['Country'] == country].dropna()
            if not country_forecast.empty:
                last_year = country_forecast['ds'].dt.year.max()
                last_value = country_forecast[country_forecast['ds'].dt.year == last_year]['yhat'].values[0]
                first_year = country_forecast['ds'].dt.year.min()
                first_value = country_forecast[country_forecast['ds'].dt.year == first_year]['yhat'].values[0]
                change_pct = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                trend = "increase 📈" if change_pct > 0 else "decrease 📉" if change_pct < 0 else "remain stable ➖"
                insights_data.append({
                    "Country": country,
                    "From": int(first_year),
                    "To": int(last_year),
                    "Trend": trend,
                    "Change (%)": f"{change_pct:.2f}%"
                })
                st.markdown(f"**{country}**: The forecast suggests a **{trend}** of approximately **{abs(change_pct):.2f}%** from {first_year} to {last_year}.")
                if trend == "increase 📈":
                    st.markdown("➡️ This upward trend may indicate economic growth or inflationary pressure, depending on the indicator. Policymakers and investors should monitor this trajectory closely.")
                elif trend == "decrease 📉":
                    st.markdown("⬇️ A downward trend could reflect positive effects like reduced inflation or negative ones like economic contraction. Further investigation is recommended.")
                else:
                    st.markdown("➖ Stability in this metric suggests a consistent economic pattern, which can be good for long-term planning and risk management.")

        # Display insights table
        if insights_data:
            st.markdown("### 📋 Summary Table")
            styled_df = pd.DataFrame(insights_data).style.format(
                {"Change (%)": "{:>}", "From": "{:>}", "To": "{:>}"}
            ).apply(
                lambda df: [
                    "background-color: #d1e7dd; font-weight: bold;" if "increase" in v 
                    else "background-color: #f8d7da; font-weight: bold;" if "decrease" in v 
                    else "background-color: #fff3cd; font-weight: bold;" for v in df
                ],
                axis=1, subset=["Trend"]
            )
            st.dataframe(styled_df, use_container_width=True)
            st.download_button(
                label="📥 Download AI Insights (CSV)",
                data=pd.DataFrame(insights_data).to_csv(index=False),
                file_name="ai_forecast_insights.csv",
                mime="text/csv"
            )
    else:
        st.info("Not enough data to forecast for the selected countries.")


# --- Download Tab --- #
with tabs[4]:
    st.subheader("Download Filtered Data")
    if not data.empty:
        st.download_button("📥 Download CSV", data.to_csv(index=False),
                           file_name=f"{selected_indicator_name}_filtered_data.csv",
                           mime="text/csv")
    else:
        st.info("No data to download.")
