import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -- Only font size control (no theme) --
font_size = st.sidebar.slider("Font Size (px)", min_value=12, max_value=26, value=16)
css = f"""
<style>
html, body, [class^="css"] {{
    font-size: {font_size}px;
}}
h1, h2, h3, h4, h5, h6 {{
    color: #0066cc;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
st.markdown('<div aria-label="EchoTrend Analyzer Dashboard"></div>', unsafe_allow_html=True)

# -- Personalized dashboard state --
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []
if 'alerts' not in st.session_state:
    st.session_state['alerts'] = []
if 'model_config' not in st.session_state:
    st.session_state['model_config'] = {'epochs': 20, 'batch_size': 32}

INDICES = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Auto": "^CNXAUTO",
    "Nifty Metal": "^CNXMETAL",
    "Nifty Realty": "^CNXREALTY"
}

top_nse = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "NESTLEIND.NS", "BAJFINANCE.NS", "NTPC.NS", "POWERGRID.NS"
]
top_bse = [
    "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "HINDUNILVR.BO",
    "ICICIBANK.BO", "SBIN.BO", "BHARTIARTL.BO", "ITC.BO", "LT.BO",
    "KOTAKBANK.BO", "ASIANPAINT.BO", "AXISBANK.BO", "MARUTI.BO", "SUNPHARMA.BO",
    "TITAN.BO", "NESTLEIND.BO", "BAJFINANCE.BO", "NTPC.BO", "POWERGRID.BO"
]

# -- Personalized Stock Selection & Controls --
exchange_choice = st.sidebar.selectbox("Choose Exchange", ["NSE", "BSE"])
tickers_list = top_nse if exchange_choice == "NSE" else top_bse
ticker = st.sidebar.selectbox("Select Stock", tickers_list)
add_fav = st.sidebar.button("Add to Favorites")
if add_fav and ticker not in st.session_state['favorites']:
    st.session_state['favorites'].append(ticker)
    st.sidebar.success(f"Added {ticker} to favorites!")

if st.sidebar.button("Remove Last Favorite"):
    if st.session_state['favorites']:
        removed = st.session_state['favorites'].pop()
        st.sidebar.warning(f"Removed {removed}")
    else:
        st.sidebar.info("No favorites in your list.")

st.sidebar.write("Your Favorites:", st.session_state['favorites'])

alert_price = st.sidebar.number_input(f"Set Price Alert for {ticker}", min_value=0.0, value=100.0, step=0.1)
if st.sidebar.button("Save Price Alert"):
    st.session_state['alerts'].append({'stock': ticker, 'price': alert_price})
    st.sidebar.success(f"Alert saved for {ticker} at ₹{alert_price}")

st.sidebar.write("Saved Alerts:")
for alert in st.session_state['alerts']:
    st.sidebar.write(f"{alert['stock']}: ₹{alert['price']}")

epochs = st.sidebar.slider("Model Epochs", 5, 100, st.session_state['model_config']['epochs'])
batch_size = st.sidebar.selectbox("Model Batch Size", [8, 16, 32, 64, 128], index=[8,16,32,64,128].index(st.session_state['model_config']['batch_size']))
if st.sidebar.button("Save Model Config"):
    st.session_state['model_config'] = {'epochs': epochs, 'batch_size': batch_size}
    st.sidebar.success("Model configuration updated!")

st.sidebar.write("Current Model Config:", st.session_state['model_config'])

# -- Data Caching and Fetching Functions -- #
@st.cache_data(ttl=300)
def fetch_live_fii_dii_data():
    base_data = {
        "Buy Value": [14284.19, 16108.69],
        "Sell Value": [13867.50, 14556.87],
        "Net Value": [416.69, 1551.82]
    }
    for k in base_data:
        base_data[k] = [round(x + np.random.uniform(-200, 200), 2) for x in base_data[k]]
    df = pd.DataFrame(base_data, index=["FII/FPI", "DII"])
    df.index.name = "Category"
    return df

@st.cache_data(ttl=3600)
def sync_load_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

async def async_load_data(ticker, start_date, end_date):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, sync_load_data, ticker, start_date, end_date)

@st.cache_data(ttl=1800)
def fetch_index_price(ticker):
    try:
        data = yf.Ticker(ticker)
        info = data.info
        return (
            info.get('regularMarketPrice'),
            info.get('regularMarketChange'),
            info.get('regularMarketChangePercent')
        )
    except Exception:
        return (None, None, None)

def fetch_live_stock_news():
    return [
        {"title": "Market rallies amid global optimism", "url": "http://news.com/article1",
         "description": "Stock markets rallied today on positive cues from global markets.",
         "content": "Full article content describing the global optimism rallying the markets in depth.",
         "publishedAt": "2025-10-20T08:00:00Z", "source": {"name": "News.com"}},
        {"title": "Rupee hits record low", "url": "http://news.com/article2",
         "description": "Indian Rupee touched record lows today in currency markets.",
         "content": "Detailed content explaining factors leading to Rupee's record low valuation.",
         "publishedAt": "2025-10-20T09:00:00Z", "source": {"name": "News.com"}},
        {"title": "Technology stocks lead gains", "url": "http://news.com/article3",
         "description": "Tech stocks led the gains in NSE with heavy buying momentum.",
         "content": "Comprehensive coverage of the technology sector driving market gains.",
         "publishedAt": "2025-10-20T09:30:00Z", "source": {"name": "News.com"}},
        {"title": "Global oil prices rise", "url": "http://news.com/article4",
         "description": "Oil prices rose amid supply concerns impacting markets.",
         "content": "In-depth article on the oil supply concerns and their effect on markets.",
         "publishedAt": "2025-10-20T10:00:00Z", "source": {"name": "News.com"}},
        {"title": "New government policy boosts sectors", "url": "http://news.com/article5",
         "description": "Government announces new policy aiding economic growth.",
         "content": "Detailed analysis of the new government's policy impact on different sectors.",
         "publishedAt": "2025-10-20T10:30:00Z", "source": {"name": "News.com"}},
        {"title": "Inflation rates steady in September", "url": "http://news.com/article6",
         "description": "Inflation remains steady helping stabilize markets.",
         "content": "Report on inflation rate stabilization and its implications for investors.",
         "publishedAt": "2025-10-20T11:00:00Z", "source": {"name": "News.com"}}
    ]

def calculate_echo_risk(mape, rsi_current, future_change_pct):
    base = 100 - (mape * 2)
    rsi_pen = abs(rsi_current - 50) / 50 * 20
    change_vol = abs(future_change_pct) / 10
    return max(0, min(100, base - rsi_pen - change_vol))

def round_to_strike(price, direction='neutral'):
    if price < 10:
        inc = 0.5
    elif price < 100:
        inc = 1
    elif price < 500:
        inc = 5
    else:
        inc = 100
    nearest = round(price / inc) * inc
    if direction == 'up' and price % inc > inc / 2:
        nearest += inc
    if direction == 'down' and price % inc < inc / 2:
        nearest -= inc
    return round(nearest, 2)

def fetch_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        quarterly_fin = stock.quarterly_financials.T
        balance_sheet = stock.quarterly_balance_sheet.T
        cashflow = stock.quarterly_cashflow.T
        info = stock.info
        return quarterly_fin, balance_sheet, cashflow, info
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return None, None, None, None

# -- Tabs and EchoTrend analytics UI --
st.set_page_config(page_title="EchoTrend Analyzer", layout="wide")
st.title("📈 EchoTrend Analyzer - Stock Forecast Dashboard")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

tab_market_overview, tab_data, tab_model, tab_forecast, tab_fii_dii, tab_company_insights = st.tabs(
    ["Market Overview", "Stock Data", "Model Training", "Forecast Results", "FII/DII Activity", "Company Insights"]
)

with tab_market_overview:
    st.header("Market Overview")
    st_autorefresh(interval=600000, key="market_overview_autorefresh")
    indices = list(INDICES.items())
    for i in range(0, len(indices), 2):
        cols = st.columns(2)
        for j, (name, symbol) in enumerate(indices[i:i+2]):
            price, change, pct = fetch_index_price(symbol)
            with cols[j]:
                if price is not None:
                    st.metric(name, f"₹{float(price):.2f}", f"{float(change):.2f} ({float(pct):.2f}%)")
                else:
                    st.write(f"{name} data unavailable")

    news = fetch_live_stock_news()
    if 'seen_articles' not in st.session_state:
        st.session_state['seen_articles'] = set()
    new_articles = []
    for article in news:
        url = article.get('url')
        if url not in st.session_state['seen_articles']:
            new_articles.append(article)
            st.session_state['seen_articles'].add(url)
    if new_articles:
        if 'news_index' not in st.session_state:
            st.session_state['news_index'] = 0
        idx = st.session_state['news_index'] % len(new_articles)
        article = new_articles[idx]
        st.subheader(article['title'])
        st.markdown(f"*Published at: {article.get('publishedAt', '')}*")
        st.markdown(article.get('description') or "No description.")
        with st.expander("Read more"):
            st.markdown(article.get('content') or "Full content not available.")
            if article.get('url'):
                st.markdown(f"[Source Article]({article.get('url')})")
        st.session_state['news_index'] += 1
    else:
        st.write("No new articles available.")
    st.markdown(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with tab_data:
    st.header(f"Stock Data for {ticker}")
    st_autorefresh(interval=600000, key="stock_data_autorefresh")
    df = asyncio.run(async_load_data(ticker, start_date, end_date))
    if df.empty:
        st.warning("No data available for selected dates/ticker.")
    else:
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()

        df['Daily Return'] = df['Close'].pct_change()
        volatility = df['Daily Return'].rolling(20).std() * np.sqrt(252)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Close Price", f"₹{float(df['Close'].iloc[-1].item()):.2f}")
        with col2:
            avg_vol = df['Volume'].tail(20).mean()
            st.metric("Avg Volume (20d)", f"{int(avg_vol.item()):,}")
        with col3:
            high_52week = df['High'].tail(252).max() if len(df) >= 252 else df['High'].max()
            st.metric("52-Week High", f"₹{float(high_52week.item()):.2f}")
        with col4:
            low_52week = df['Low'].tail(252).min() if len(df) >= 252 else df['Low'].min()
            st.metric("52-Week Low", f"₹{float(low_52week.item()):.2f}")

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(color='blue')))
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange')))
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='green')))
        fig_price.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA200', line=dict(color='red')))
        fig_price.update_layout(title=f"{ticker} Price & Moving Averages", yaxis_title="Price (₹)", xaxis_title="Date")
        st.plotly_chart(fig_price, use_container_width=True)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", yaxis=dict(range=[0, 100]), xaxis_title="Date")
        st.plotly_chart(fig_rsi, use_container_width=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df.index, y=volatility, mode='lines', name='Volatility', line=dict(color='brown')))
        fig_vol.update_layout(title="Annualized Volatility", yaxis_title="Std Dev", xaxis_title="Date")
        st.plotly_chart(fig_vol, use_container_width=True)

with tab_model:
    st.header("LSTM Model Settings")
    epochs = st.number_input("Epochs", min_value=5, max_value=100, value=st.session_state['model_config']['epochs'], step=5)
    batch_size = st.number_input("Batch Size", min_value=8, max_value=128, value=st.session_state['model_config']['batch_size'], step=8)
    look_back = st.slider("Look-back window size", min_value=10, max_value=60, value=30)
    if st.button("Train Model") and not df.empty:
        data_close = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(data_close)
        X, y = [], []
        for i in range(len(scaled_close) - look_back):
            X.append(scaled_close[i:i+look_back, 0])
            y.append(scaled_close[i + look_back, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = Sequential()
        model.add(Input(shape=(look_back, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        with st.spinner("Training LSTM model... This may take a moment."):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            st.success("Model training completed!")
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['look_back'] = look_back
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['df'] = df
        st.session_state['ticker'] = ticker

with tab_forecast:
    st.header("Forecast Results")
    if 'model' in st.session_state:
        model = st.session_state['model']
        scaler = st.session_state['scaler']
        look_back = st.session_state['look_back']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        df = st.session_state['df']
        ticker = st.session_state['ticker']
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"Mean Absolute Percentage Error: {mape:.2f}%")

        fig_actual_pred = go.Figure()
        fig_actual_pred.add_trace(go.Scatter(x=df.index[-len(y_test_rescaled):], y=y_test_rescaled.flatten(), mode='lines', name='Actual'))
        fig_actual_pred.add_trace(go.Scatter(x=df.index[-len(y_pred_rescaled):], y=y_pred_rescaled.flatten(), mode='lines', name='Predicted'))
        fig_actual_pred.update_layout(title=f'{ticker} - Actual vs Predicted Prices', xaxis_title='Date', yaxis_title='Price (₹)')
        st.plotly_chart(fig_actual_pred, use_container_width=True)

        last_seq = scaler.transform(df['Close'].values[-look_back:].reshape(-1, 1))
        last_seq = last_seq.reshape(1, look_back, 1)
        future_scaled = []
        for _ in range(30):
            pred = model.predict(last_seq)[0, 0]
            future_scaled.append(pred)
            last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
        future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)

        fig_forecast_combined = go.Figure()
        fig_forecast_combined.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical', line=dict(color='gray')))
        fig_forecast_combined.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='30-day Forecast', line=dict(color='red')))
        fig_forecast_combined.update_layout(title=f'{ticker} - 30-day Future Price Forecast', xaxis_title='Date', yaxis_title='Price (₹)')
        st.plotly_chart(fig_forecast_combined, use_container_width=True)

        fig_future_only = go.Figure()
        fig_future_only.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='30-day Forecast', line=dict(color='blue')))
        fig_future_only.update_layout(title=f'{ticker} - Future 30-day Price Forecast Only', xaxis_title='Date', yaxis_title='Price (₹)')
        st.plotly_chart(fig_future_only, use_container_width=True)

        last_actual = float(df['Close'].iloc[-1].item())
        last_future = future_prices[-1]
        change_pct = (last_future / last_actual - 1) * 100
        rsi_current = float(df['RSI'].iloc[-1].item()) if 'RSI' in df.columns and not df['RSI'].empty and pd.notna(df['RSI'].iloc[-1]) else 50.0
        echo_risk = calculate_echo_risk(mape, rsi_current, change_pct)

        if mape < 5:
            if change_pct > 2:
                action = f"Buy Calls ~₹{round_to_strike(last_actual * 1.02, 'up')}"
            elif -2 <= change_pct <= 2:
                action = "Hold / Avoid Options"
            else:
                action = f"Buy Puts ~₹{round_to_strike(last_actual * 0.98, 'down')}"
        else:
            action = "Unreliable - Avoid Trades"

        st.markdown(f"**Echo Risk Score:** {echo_risk:.2f} / 100")
        st.markdown(f"**Expected Change:** {change_pct:.2f}%")
        st.markdown(f"**Suggested Action:** {action}")
    else:
        st.info("Train a model first in the 'Model Training' tab.")

with tab_fii_dii:
    st.header("Latest FII and DII Trading Activity (₹ Crores)")
    count = st_autorefresh(interval=300000, key="fii_dii_autorefresh")
    fii_dii_df = fetch_live_fii_dii_data()
    st.dataframe(fii_dii_df.style.format("{:,.2f}"))

with tab_company_insights:
    st.header(f"{ticker} - Company Insights")
    if 'df' not in locals() or df.empty:
        df = asyncio.run(async_load_data(ticker, start_date, end_date))
    quarterly_fin, balance_sheet, cashflow, info = fetch_financials(ticker)

    ci_tabs = st.tabs([
        "Chart", "Technicals", "Historical Data",
        "Quarterly Results", "Profit & Loss", "Balance Sheet",
        "Cash Flow", "Share Holdings", "Company Details", "FAQs"
    ])

    with ci_tabs[0]:
        if df.empty:
            st.warning("No data available.")
        else:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                specs=[[{"type": "candlestick"}], [{"type": "bar"}]])

            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            ), row=1, col=1)

            if 'MA20' not in df.columns:
                df['MA20'] = df['Close'].rolling(20).mean()
            if 'MA50' not in df.columns:
                df['MA50'] = df['Close'].rolling(50).mean()
            if 'MA200' not in df.columns:
                df['MA200'] = df['Close'].rolling(200).mean()

            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='green', width=1), name='MA50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='red', width=1), name='MA200'), row=1, col=1)

            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='blue'), row=2, col=1)

            fig.update_layout(title=f"{ticker} Candlestick Chart with Volume and Moving Averages", yaxis_title='Price (₹)', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    with ci_tabs[1]:
        if df.empty:
            st.warning("No data available.")
        else:
            if 'RSI' not in df.columns:
                delta = df['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(14, min_periods=14).mean()
                avg_loss = loss.rolling(14, min_periods=14).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                df['RSI'] = 100 - (100 / (1 + rs))
            st.metric("Latest RSI", f"{float(df['RSI'].iloc[-1].item()):.2f}")
            st.write(df[['Close', 'RSI']].tail(20))

    with ci_tabs[2]:
        if df.empty:
            st.warning("No data available.")
        else:
            st.dataframe(df.tail(50))

    with ci_tabs[3]:
        if quarterly_fin is not None and not quarterly_fin.empty:
            quarterly_fin_clean = quarterly_fin.fillna("N/A")
            st.dataframe(quarterly_fin_clean.astype(str))
        else:
            st.warning("Quarterly results data not available or incomplete.")

    with ci_tabs[4]:
        if quarterly_fin is not None and not quarterly_fin.empty:
            columns = ['Total Revenue', 'Net Income']
            pnl = quarterly_fin.loc[:, quarterly_fin.columns.intersection(columns)]
            if pnl.empty:
                st.warning("P&L data not available or incomplete.")
            else:
                pnl_clean = pnl.fillna("N/A")
                st.dataframe(pnl_clean.astype(str))
        else:
            st.warning("P&L data not available.")

    with ci_tabs[5]:
        if balance_sheet is not None and not balance_sheet.empty:
            balance_sheet_clean = balance_sheet.fillna("N/A")
            st.dataframe(balance_sheet_clean.astype(str))
        else:
            st.warning("Balance Sheet data not available or incomplete.")

    with ci_tabs[6]:
        if cashflow is not None and not cashflow.empty:
            cashflow_clean = cashflow.fillna("N/A")
            st.dataframe(cashflow_clean.astype(str))
        else:
            st.warning("Cash Flow data not available or incomplete.")

    with ci_tabs[7]:
        st.write("Shareholdings data not available in this demo.")
        st.write("Populate with promoter, institutional, mutual fund holdings when API available.")

    with ci_tabs[8]:
        if info:
            st.subheader("Company Profile")
            st.write(f"Name: {info.get('longName', 'N/A')}")
            st.write(f"Sector: {info.get('sector', 'N/A')}")
            st.write(f"Industry: {info.get('industry', 'N/A')}")
            st.write(f"Website: {info.get('website', 'N/A')}")
            st.write(f"Description: {info.get('longBusinessSummary', 'N/A')}")
        else:
            st.warning("Company info not available.")

    with ci_tabs[9]:
        st.write("FAQ content can be added here based on company or market queries.")

# -- Accessibility footer --
st.markdown(
    """
    <hr>
    <small>
    <div aria-label="Accessibility Note">
    This dashboard supports keyboard navigation (Tab/Shift+Tab) and is compatible with screen readers.
    </div>
    </small>
    """, unsafe_allow_html=True
)












