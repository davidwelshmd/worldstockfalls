import datetime
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. Page Configuration
st.set_page_config(
    page_title="Global Market Index Performance",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Comprehensive Global Indices Dictionary
@st.cache_data
def get_indices_dict():
    return {
        # --- NORTH AMERICA ---
        "^GSPC": "S&P 500 (USA)",
        "^DJI": "Dow Jones Industrial Average (USA)",
        "^IXIC": "NASDAQ Composite (USA)",
        "^RUT": "Russell 2000 (USA)",
        "^GSPTSE": "S&P/TSX Composite (Canada)",
        "^MXX": "S&P/BMV IPC (Mexico)",
        # --- SOUTH & CENTRAL AMERICA ---
        "^BVSP": "IBOVESPA (Brazil)",
        "^MERV": "S&P MERVAL (Argentina)",
        "^IPSA": "S&P/CLX IPSA (Chile)",
        "COLCAP.CB": "MSCI COLCAP (Colombia)",
        "^SPBLPGPT": "S&P/BVL Peru General (Peru)",
        # --- WESTERN & NORTHERN EUROPE ---
        "^FTSE": "FTSE 100 (UK)",
        "^GDAXI": "DAX Performance Index (Germany)",
        "^FCHI": "CAC 40 (France)",
        "^STOXX50E": "Euro Stoxx 50 (Eurozone)",
        "^AEX": "AEX Index (Netherlands)",
        "^SSMI": "Swiss Market Index (Switzerland)",
        "^BFX": "BEL 20 (Belgium)",
        "^OMXSPI": "OMX Stockholm 30 (Sweden)",
        "^OSEAX": "Oslo Børs All-Share (Norway)",
        "OMXH25.HE": "OMX Helsinki 25 (Finland)",
        # --- SOUTHERN & EASTERN EUROPE ---
        "^IBEX": "IBEX 35 (Spain)",
        "FTSEMIB.MI": "FTSE MIB (Italy)",
        "^PSI50": "PSI (Portugal)",
        "^WIG20": "WIG20 (Poland)",
        "XU100.IS": "BIST 100 (Turkey)",
        "^ATX": "Austrian Traded Index (Austria)",
        # --- MIDDLE EAST & AFRICA ---
        "^TA125": "TA-125 (Israel)",
        "^TASI.SR": "Tadawul All Share (Saudi Arabia)",
        "^DFMGI": "DFM General Index (Dubai)",
        "PASI.QA": "QE Index (Qatar)",
        "^J203.JO": "JSE All Share (South Africa)",
        "^EGX30": "EGX 30 (Egypt)",
        # --- ASIA ---
        "^N225": "Nikkei 225 (Japan)",
        "^HSI": "Hang Seng Index (Hong Kong)",
        "000001.SS": "SSE Composite Index (China)",
        "399001.SZ": "Shenzhen Component (China)",
        "^BSESN": "S&P BSE SENSEX (India)",
        "^NSEI": "NIFTY 50 (India)",
        "^KS11": "KOSPI Composite (South Korea)",
        "^TWII": "TSEC Weighted Index (Taiwan)",
        "^STI": "Straits Times Index (Singapore)",
        "^KLSE": "FTSE Bursa Malaysia KLCI (Malaysia)",
        "^JKSE": "Jakarta Composite (Indonesia)",
        "^PSEI.PS": "PSEi Composite (Philippines)",
        "^SET.BK": "SET Index (Thailand)",
        "VNINDEX.VN": "VN-Index (Vietnam)",
        # --- OCEANIA ---
        "^AXJO": "S&P/ASX 200 (Australia)",
        "^NZ50": "S&P/NZX 50 (New Zealand)",
    }


# 3. Data Processing Engine
@st.cache_data(ttl=3600)
def fetch_and_calculate_performance(indices):
    records = []
    today = datetime.date.today()
    three_years_ago = today - datetime.timedelta(days=3 * 365)
    one_year_ago = today - datetime.timedelta(days=365)

    tickers_list = list(indices.keys())

    # Download via single batch request to avoid API throttling blocks
    try:
        raw_data = yf.download(
            tickers=tickers_list,
            start=three_years_ago,
            end=today,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw_data.empty:
        return pd.DataFrame()

    # Process metrics
    for ticker, name in indices.items():
        try:
            # Check if multi-index contains ticker frame safely
            if ticker not in raw_data.columns.levels[0]:
                continue

            # Isolate individual series using cross-sections
            df_ticker = raw_data[ticker]["Close"].dropna()

            if df_ticker.empty or len(df_ticker) < 5:
                continue

            current_val = df_ticker.iloc[-1]

            # Approximate historical reference markers
            idx_12m = df_ticker.index.searchsorted(pd.Timestamp(one_year_ago))
            idx_3yr = df_ticker.index.searchsorted(pd.Timestamp(three_years_ago))

            # Bound constraints protection
            idx_12m = min(idx_12m, len(df_ticker) - 1)
            idx_3yr = min(idx_3yr, len(df_ticker) - 1)

            val_12m = df_ticker.iloc[idx_12m]
            val_3yr = df_ticker.iloc[idx_3yr]

            # Performance Math calculations
            perf_12m = ((current_val - val_12m) / val_12m) * 100
            perf_3yr = ((current_val - val_3yr) / val_3yr) * 100

            records.append(
                {
                    "Ticker": ticker,
                    "Index Name": name,
                    "Current Level": round(current_val, 2),
                    "12-Month Return (%)": round(perf_12m, 2),
                    "3-Year Return (%)": round(perf_3yr, 2),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(records)


# 4. Presentation UI Setup
st.title("导 Comprehensive Global Market Indices Decline Analyzer")
st.markdown(
    "Analyze and isolate stock market indices that have experienced the steepest corrections globally over 12-month and 3-year windows."
)

indices_dict = get_indices_dict()

with st.spinner("Downloading global market data in a single batch..."):
    df_metrics = fetch_and_calculate_performance(indices_dict)

if not df_metrics.empty:
    # Sidebar Sorting Selection Controls
    st.sidebar.header("Ranking Configuration")
    sort_horizon = st.sidebar.selectbox(
        "Primary Sort Benchmark",
        options=["12-Month Return (%)", "3-Year Return (%)"],
        index=0,
    )

    # Sort from worst performing (biggest fall) to best performing
    df_sorted = df_metrics.sort_values(by=sort_horizon, ascending=True)

    # Human-readable formatting function
    def format_dataframe(df):
        styled_df = df.copy()
        styled_df["12-Month Return (%)"] = styled_df["12-Month Return (%)"].map(
            "{:+.2f}%".format
        )
        styled_df["3-Year Return (%)"] = styled_df["3-Year Return (%)"].map(
            "{:+.2f}%".format
        )
        styled_df["Current Level"] = styled_df["Current Level"].map(
            "{:,.2f}".format
        )
        return styled_df

    # Render Main Data Table Grid
    st.subheader(f"Global Indices Ranked by Performance ({sort_horizon})")
    st.markdown(
        f"**Successfully retrieved {len(df_sorted)} out of {len(indices_dict)} global indices.** "
        f"Top entries reflect the largest falls."
    )

    st.dataframe(
        format_dataframe(df_sorted),
        hide_index=True,
        use_container_width=True,
    )

    # High-Risk Flags UI Component
    st.subheader("⚠️ Critical Capital Corrections")
    col1, col2 = st.columns(2)

    with col1:
        worst_12m = df_metrics.sort_values(by="12-Month Return (%)").iloc[0]
        st.metric(
            label=f"12-Month Maximum Drawdown ({worst_12m['Index Name']})",
            value=f"{worst_12m['12-Month Return (%)']}%",
            delta=f"{worst_12m['Ticker']}",
            delta_color="off",
        )

    with col2:
        worst_3yr = df_metrics.sort_values(by="3-Year Return (%)").iloc[0]
        st.metric(
            label=f"3-Year Maximum Drawdown ({worst_3yr['Index Name']})",
            value=f"{worst_3yr['3-Year Return (%)']}%",
            delta=f"{worst_3yr['Ticker']}",
            delta_color="off",
        )
else:
    st.error(
        "Yahoo Finance timed out or rate-limited the connection. Please refresh the page in a few moments."
    )
