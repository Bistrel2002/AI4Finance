# 📈 AI4Finance – Stock Price Prediction Application with Machine Learning

AI4Finance is a comprehensive Machine Learning project aimed at predicting **stock price trends** (rise or fall) from **historical market data**.  
This project is ideal for demonstrating skills in: financial data processing, ML modeling, simple API deployment, and user interface creation.

---

## 🧠 Learning Objectives

- Collect and explore **real-time financial data**
- Perform relevant **feature engineering** (stock indicators)
- Build and train a **binary classification model**
- Create an **interactive prediction interface with Streamlit**
- **Deploy** an application usable by recruiters or investors

---

## 🛠️ Technologies Used

| Category          | Tools / Libraries                        |
|------------------|------------------------------------------|
| Language         | Python 3.x                               |
| Data             | `yfinance`, `pandas`, `numpy`            |
| Visualization    | `matplotlib`, `plotly`, `seaborn`        |
| Modeling         | `scikit-learn`, `xgboost`, `joblib`      |
| Web Interface    | `streamlit`                              |
| Deployment       | `Render`, `Streamlit Cloud`, `HuggingFace Spaces` |

---

## 📁 Project Structure

```
AI4Finance/
├── app/
│   └── app.py                    # Streamlit Application
├── data/                         # CSV data downloaded from yfinance
├── model/
│   └── model.pkl                # Saved trained model
├── notebooks/
│   └── EDA.ipynb                # Notebook for analysis
├── main.py                      # Principal
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Installation and Usage

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd AI4Finance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**
   ```bash
   streamlit run app/app.py
   ```
   Here is what I was saying and they desided to put me into prison that is not fair at all

---

## 📊 Features

- **Automatic download** of stock data via yfinance
- **Exploratory analysis** with interactive visualizations
- **Feature engineering** with technical indicators (RSI, MACD, etc.)
- **Prediction model** trained on historical data
- **Intuitive user interface** for real-time predictions

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features

---

## 📝 License

This project is under MIT license. See the LICENSE file for more details.
