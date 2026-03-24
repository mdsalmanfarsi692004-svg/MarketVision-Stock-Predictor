import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Stock Predictor Pro", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    
    .main-title {
        text-align: center; font-size: 3.5rem; font-weight: 900; margin-bottom: 0px;
    }
    .gradient-text {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FFD700, #FF8C00);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
    }
    .sub-title { text-align: center; font-size: 1.2rem; color: #a0aec0; margin-top: -10px; margin-bottom: 30px; }
    
    button[kind="primary"] {
        background: linear-gradient(90deg, #FF8C00 0%, #FF4500 100%);
        border: none; border-radius: 8px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.4);
    }
    button[kind="primary"]:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# --- CACHED DATA FETCHING (Now with Expiry to prevent dead caching) ---
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# --- HERO SECTION ---
st.markdown('<div class="main-title">📈 <span class="gradient-text">AI Stock Trend Predictor</span> 📈</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Deep Learning Engine (LSTM) & Financial Dashboard</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("<h2 style='text-align: center;'>⚙️ Trading Desk</h2>", unsafe_allow_html=True)
    
    # User Inputs with auto-strip to prevent space errors
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS)", value="RELIANCE.NS").strip().upper()
    
    col1, col2 = st.columns(2)
    with col1:
        # Automatically sets start date to 2 years ago to avoid IPO date issues
        two_years_ago = datetime.date.today() - datetime.timedelta(days=730)
        start_date = st.date_input("Start Date", two_years_ago)
    with col2:
        end_date = st.date_input("End Date", datetime.date.today())
        
    predict_btn = st.button("🧠 Run AI Prediction (LSTM)", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("### 💡 How it works:")
    st.markdown("1. Fetches live market data.\n2. Calculates Moving Averages.\n3. Trains a **Deep Neural Network (LSTM)**.\n4. Predicts tomorrow's trend.")

# --- MAIN DASHBOARD & LOGIC ---
if ticker:
    try:
        data = load_data(ticker, start_date, end_date)
        
        if data.empty:
            st.error(f"⚠️ No data found for {ticker}. Please check the ticker symbol (e.g., use .NS for Indian stocks). Also, ensure the Start Date is after the company's IPO.")
        else:
            # Calculate Indicators
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()
            
            # --- METRICS UI ---
            current_price = round(data['Close'].iloc[-1].item(), 2)
            previous_price = round(data['Close'].iloc[-2].item(), 2)
            price_change = round(current_price - previous_price, 2)
            change_percent = round((price_change / previous_price) * 100, 2)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Live Market Price", f"${current_price}", f"{price_change} ({change_percent}%)")
            m2.metric("50-Day Moving Avg", f"${round(data['MA_50'].iloc[-1].item(), 2)}")
            m3.metric("Total Trading Volume", f"{int(data['Volume'].iloc[-1].item()):,}")
            
            # --- INTERACTIVE CHART ---
            st.markdown("### 📊 Interactive Technical Chart")
            fig = go.Figure()
            # Candlestick
            fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Market Data'))
            # Moving Averages
            fig.add_trace(go.Scatter(x=data['Date'], y=data['MA_50'], line=dict(color='orange', width=2), name='50-Day MA'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['MA_200'], line=dict(color='blue', width=2), name='200-Day MA'))
            
            fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # --- AI PREDICTION ENGINE (LSTM) ---
            if predict_btn:
                st.divider()
                st.markdown("### 🤖 Deep Learning Prediction (LSTM)")
                
                with st.spinner("Training LSTM Neural Network on historical data... Please wait (Approx 15-30 secs)..."):

                    # 1. Prepare Data
                    dataset = data['Close'].values.reshape(-1, 1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(dataset)
                    
                    # Use last 60 days to predict next day
                    train_len = int(np.ceil(len(dataset) * 0.8))
                    train_data = scaled_data[0:int(train_len), :]
                    
                    x_train, y_train = [], []
                    for i in range(60, len(train_data)):
                        x_train.append(train_data[i-60:i, 0])
                        y_train.append(train_data[i, 0])
                        
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    
                    # 2. Build LSTM Model
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(25))
                    model.add(Dense(1))
                    
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    # 3. Train Model (Fast training for web app)
                    model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=0)
                    
                    # 4. Predict Next Day
                    last_60_days = scaled_data[-60:]
                    X_test = np.array([last_60_days])
                    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                    
                    pred_price_scaled = model.predict(X_test)
                    predicted_price = float(scaler.inverse_transform(pred_price_scaled)[0][0])
                    
                    # --- AI RESULT UI ---
                    st.success("✅ Neural Network Training Complete!")
                    
                    trend_color = "#10b981" if predicted_price > current_price else "#ef4444"
                    trend_icon = "📈 UPWARD" if predicted_price > current_price else "📉 DOWNWARD"
                    
                    # NOTE: HTML Block with NO INDENTATION
                    html_result = f"""
<div style="background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid #334155; border-left: 5px solid {trend_color}; border-radius: 10px; padding: 25px; text-align: center; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);">
    <div style="color: #94a3b8; font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">LSTM Predicted Price for Next Trading Day</div>
    <div style="color: #f8fafc; font-size: 48px; font-weight: 900; margin-bottom: 5px;">${predicted_price:.2f}</div>
    <div style="color: {trend_color}; font-size: 18px; font-weight: 600;">Expected Trend: {trend_icon}</div>
</div>
"""
                    st.markdown(html_result, unsafe_allow_html=True)
                    st.caption("Disclaimer: This is an AI prediction based on historical data. Do not use this for actual financial trading.")

    except Exception as e:
        st.error(f"Error fetching data: {e}")