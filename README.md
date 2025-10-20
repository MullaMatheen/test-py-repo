# 📈 EchoTrend Analyzer

[![GitHub](https://img.shields.io/badge/GitHub-MullaMatheen-black?logo=github)](https://github.com/MullaMatheen)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Abdul%20Matheen%20Mulla-blue?logo=linkedin)](https://www.linkedin.com/in/mulla-abdul-matheen-013970227/))
[![Email](https://img.shields.io/badge/Email-matheenmulla786%40gmail.com-red?logo=gmail)](mailto:matheenmulla786@gmail.com)

**EchoTrend Analyzer** is an interactive **Streamlit dashboard** for stock market forecasting and analysis.
It uses **LSTM (Long Short-Term Memory)** neural networks to predict stock prices, visualize trends, and suggest possible trading actions based on risk and accuracy metrics.

---

## 🚀 Features

* **Live stock data** fetched using `yfinance`
* Supports both **NSE** and **BSE** exchanges
* **Top 20 stocks** preloaded with an option for custom tickers
* Computes key **technical indicators**:

  * SMA (Simple Moving Average)
  * EMA (Exponential Moving Average)
  * RSI (Relative Strength Index)
  * ATR (Average True Range)
* Trains a **neural network (LSTM)** to learn stock patterns
* Predicts **30 days of future stock prices**
* Generates:

  * 📊 Historical vs Predicted graph
  * 🔮 30-day forecast chart
* Displays:

  * **MAPE (Mean Absolute Percentage Error)**
  * **Echo Risk Score**
  * **Suggested Action (Buy/Hold/Sell)**

---

## 🧠 How It Works

1. Downloads stock data between selected start and end dates.
2. Calculates technical indicators and scales them with `MinMaxScaler`.
3. Builds a two-layer LSTM model using TensorFlow/Keras.
4. Predicts both historical and future prices.
5. Evaluates accuracy with **MAE** and **MAPE**.
6. Suggests actions based on forecasted trend and accuracy.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MullaMatheen/EchoTrend-Analyzer.git
cd EchoTrend-Analyzer
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit yfinance pandas numpy matplotlib scikit-learn tensorflow
```

---

## ▶️ Running the App

Start the dashboard:

```bash
streamlit run echotrend_streamlit.py
```

Then open the local URL (usually [http://localhost:8501]) in your browser.

---

## 📂 Project Structure

```
EchoTrend-Analyzer/
│
├── echotrend_streamlit.py      # Main Streamlit app
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
└── (optional) /data/           # Cached stock data or logs
```

---

## 📊 Example Output

* Historical vs Predicted chart with ±MAE shading
* 30-day forecast line for trend visualization
* Metrics: MAPE %, Echo Risk score, Expected price change
* Suggested action (Buy Calls / Buy Puts / Hold)

---

## 👨‍💻 Author

**Abdul Matheen Mulla**
🎓 B.Tech in Computer Science, Keshav Memorial Engineering College
📧 Email: [matheenmulla786@gmail.com](mailto:matheenmulla786@gmail.com)
🔗 [GitHub](https://github.com/MullaMatheen) | [LinkedIn](https://www.linkedin.com/in/abdul-matheen-mulla-013970227)

---

## 🧾 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

### 💡 Future Improvements

* Add ARIMA and Prophet model options
* Include news sentiment analysis
* Deploy on Streamlit Cloud or Hugging Face Spaces

