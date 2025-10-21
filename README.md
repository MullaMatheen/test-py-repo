# 📈 EchoTrend Analyzer

EchoTrend Analyzer is an advanced, interactive Streamlit dashboard for analyzing **Indian stock markets** (NSE/BSE), forecasting prices using **LSTM neural networks**, and providing actionable trading insights.  
It combines real-time data from Yahoo Finance with technical indicators, machine learning models, and an intuitive UI.

***

## 🌟 Features

- 🎨 **Custom UI Controls:** Light/Dark theme and adjustable font size.  
- ❤️ **Personalized Dashboard:** Save favorites, set price alerts, and export configurations.  
- 📊 **Stock Data Analytics:** OHLCV, moving averages (MA20/50/200), RSI, volatility, and CSV export.  
- 📰 **Market Overview:** Real-time indices, auto-refreshing news dashboard.  
- 🤖 **LSTM Forecasting:** Configure, train, and visualize 30‑day price predictions.  
- ⚡ **Echo Risk Score:** Smart trade signals (Buy/Hold/Sell) using MAPE, RSI, and predicted changes.  
- 💼 **Institutional Insights:** Live FII/DII trading activity in ₹ Crores.  
- 🧾 **Company Fundamentals:** Access quarterly results, P&L, balance sheets, and company details.  
- ♿ **Accessibility:** Screen reader ready and keyboard-navigation friendly.

***

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MullaMatheen/echotrend-analyzer.git
cd echotrend-analyzer
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the App
```bash
streamlit run echotrend_analyzer.py
```

***

## 🧠 Usage Instructions

- Select your preferred **exchange (NSE/BSE)** and a **stock** from the sidebar.  
- Customize **model parameters** (epochs, batch size, look-back window).  
- Explore multiple tabs:
  - *Market Overview* – Index metrics & financial news  
  - *Stock Data* – Prices, RSI, volatility, CSV download  
  - *Model Training* – Setup and train LSTM  
  - *Forecast Results* – 30-day predictions and risk insights  
  - *FII/DII Activity* – Institutional trading table  
  - *Company Insights* – Charts, technicals, financials, P&L, balance sheet  

***

## 📁 Project Structure

```
echotrend-analyzer/
│
├── echotrend_analyzer.py     # Main Streamlit Application
├── requirements.txt          # Python dependencies
├── README.md                 # Project Documentation
└── assets/                   # (Optional) icons, images, related scripts
```

***

## 🧩 Requirements

This project requires:
- Python ≥ 3.8
- streamlit  
- pandas  
- numpy  
- yfinance  
- requests  
- plotly  
- scikit-learn  
- tensorflow  
- streamlit-autorefresh  

Install them easily using:
```bash
pip install -r requirements.txt
```

***

## 📬 Contact

**Author:** Mulla Matheen  
**Email:** [matheenmulla786@gmail.com](mailto:matheenmulla786@gmail.com)  
**GitHub:** [MullaMatheen](https://github.com/MullaMatheen)  

***

## ⚖️ License

This project is open-source under the **MIT License**.  
Feel free to use, modify, and distribute with attribution.

***

## 🙌 Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) API (`yfinance`)  
- [Streamlit](https://streamlit.io) for interactive UI  
- [Plotly](https://plotly.com/) for advanced visualizations  
- [TensorFlow](https://www.tensorflow.org/) for deep learning backend  

***

### 💡 Tip:
If you like the project, give it a ⭐️ on GitHub to support continued improvements!

***

Would you like me to also generate a **`requirements.txt`** file optimized for this project so you can directly upload both to GitHub?

