# 📈 MarketVision ML - Deep Learning Stock Predictor

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000.svg)
![yfinance](https://img.shields.io/badge/API-yfinance-blue.svg)

## 🚀 Project Overview
MarketVision ML is an advanced Time-Series forecasting dashboard. It fetches live market data directly from Yahoo Finance and utilizes a Deep Learning Long Short-Term Memory (LSTM) Neural Network to predict future stock price movements.

## Live App Link:

https://marketvision-stock-predictor.streamlit.app/

## Screenshots

## Main Dashboard
<img width="1919" height="923" alt="Screenshot 2026-03-24 114348" src="https://github.com/user-attachments/assets/d7af271d-a470-4b7e-9ba8-2d3a8625c043" />

## Tesla Prediction
<img width="1912" height="913" alt="Screenshot 2026-03-24 113649" src="https://github.com/user-attachments/assets/a5e8e0ef-2878-4012-8f43-33997ee1f3b7" />

## Apple Prediction
<img width="1912" height="913" alt="Screenshot 2026-03-24 113525" src="https://github.com/user-attachments/assets/188888d3-b432-45da-a8d1-6a506279e255" />

*Developed as part of the AI & ML Internship at Elevate Labs.*

## 🧠 System Architecture & Methodology
1. **Data Ingestion:** Real-time historical data scraping using the `yfinance` API.
2. **Feature Engineering:** Calculates 50-Day and 200-Day Moving Averages.
3. **Data Normalization:** Scales financial data using `MinMaxScaler`.
4. **LSTM Modeling:** Utilizes a multi-layer LSTM architecture designed to retain long-term sequential memory.

## 🔥 Key Features
* **Deep Learning Engine:** Robust predictive modeling for accurate trend forecasting.
* **Live Market Metrics:** Displays real-time prices, volume, and percentage changes.
* **Interactive Technical Charting:** Features responsive candlestick charts using Plotly.

## ⚙️ Installation & Usage
git clone https://github.com/your-username/MarketVision-Stock-Predictor.git
cd MarketVision-Stock-Predictor
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
**Md Salman Farsi**
* **Role:** AI & ML Intern @ Elevate Labs | B.Tech CS (AI & ML)
* **Portfolio:** [mdsalmanfarsi.io](https://mdsalmanfarsi.io)
* **Email:** [mdsalmanfarsi692004@gmail.com](mailto:mdsalmanfarsi692004@gmail.com)
* **GitHub:** [Your GitHub Profile Link]
* **LinkedIn:** [Your LinkedIn Profile Link]
