# 📡 ChurnSight — Telco Customer Churn Prediction (Flask App)

A production-ready Flask web application that serves a trained XGBoost machine
learning model for predicting telecom customer churn.

---

## 📂 Project Structure

```
churn-flask-app/
│
├── app.py                  ← Flask backend (routes, preprocessing, prediction)
├── Procfile                ← For Render / Railway deployment
├── requirements.txt        ← Python dependencies
├── README.md
│
├── models/                 ← ⭐ Place your trained model files here
│   ├── churn_model.pkl     ← Trained XGBoost model (from notebook)
│   └── scaler.pkl          ← Fitted StandardScaler (from notebook)
│
├── templates/
│   └── index.html          ← Bootstrap + custom HTML frontend
│
└── static/
    └── style.css           ← Dark-theme custom stylesheet
```

---

## ⚡ Quick Setup (Local)

### Step 1 — Clone / download the project

```bash
git clone https://github.com/your-username/churn-flask-app.git
cd churn-flask-app
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Copy your trained model files

From your Jupyter Notebook (Step 10), copy the two saved files:

```
churn_model.pkl   →   models/churn_model.pkl
scaler.pkl        →   models/scaler.pkl
```

If your notebook saved them to the current directory, run:

```bash
# macOS / Linux
cp /path/to/notebook/churn_model.pkl models/
cp /path/to/notebook/scaler.pkl      models/

# Windows
copy C:\path\to\notebook\churn_model.pkl models\
copy C:\path\to\notebook\scaler.pkl      models\
```

### Step 5 — Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🌐 API Endpoints

| Method | Endpoint      | Description                          |
|--------|---------------|--------------------------------------|
| GET    | `/`           | Prediction form (HTML UI)            |
| POST   | `/predict`    | Predict churn (form or JSON)         |
| GET    | `/health`     | API health check                     |
| GET    | `/api/sample` | Returns a sample JSON request body   |

---

## 🧪 Testing with Postman

### High-Risk Customer (should predict Churn)

**Method:** `POST`  
**URL:** `http://localhost:5000/predict`  
**Headers:**
```
Content-Type: application/json
Accept:       application/json
```

**Body (raw JSON):**
```json
{
  "gender"          : "Male",
  "SeniorCitizen"   : 0,
  "Partner"         : "No",
  "Dependents"      : "No",
  "tenure"          : 2,
  "PhoneService"    : "Yes",
  "MultipleLines"   : "No",
  "InternetService" : "Fiber optic",
  "OnlineSecurity"  : "No",
  "OnlineBackup"    : "No",
  "DeviceProtection": "No",
  "TechSupport"     : "No",
  "StreamingTV"     : "No",
  "StreamingMovies" : "No",
  "Contract"        : "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod"   : "Electronic check",
  "MonthlyCharges"  : 79.85,
  "TotalCharges"    : 159.70
}
```

**Expected Response:**
```json
{
  "status": "success",
  "result": {
    "prediction"        : "Churn",
    "churn_probability" : 78.34,
    "stay_probability"  : 21.66,
    "risk_level"        : "High",
    "risk_color"        : "danger",
    "will_churn"        : true,
    "message"           : "⚠️ This customer is very likely to churn. Immediate retention action is strongly recommended."
  }
}
```

### Low-Risk Customer (should predict No Churn)

```json
{
  "gender"          : "Female",
  "SeniorCitizen"   : 0,
  "Partner"         : "Yes",
  "Dependents"      : "Yes",
  "tenure"          : 60,
  "PhoneService"    : "Yes",
  "MultipleLines"   : "Yes",
  "InternetService" : "DSL",
  "OnlineSecurity"  : "Yes",
  "OnlineBackup"    : "Yes",
  "DeviceProtection": "Yes",
  "TechSupport"     : "Yes",
  "StreamingTV"     : "Yes",
  "StreamingMovies" : "Yes",
  "Contract"        : "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod"   : "Bank transfer (automatic)",
  "MonthlyCharges"  : 89.10,
  "TotalCharges"    : 5346.00
}
```

### cURL Example

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":2,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":79.85,"TotalCharges":159.70}'
```

---

## 🚀 Deployment Guide

### Option 1 — Render (Free Tier)

1. Push your project to GitHub (make sure `models/` folder with `.pkl` files is included).
2. Go to [render.com](https://render.com) → **New** → **Web Service**.
3. Connect your GitHub repository.
4. Fill in:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:**  `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment:** Python 3
5. Click **Deploy**.
6. Your app will be live at `https://your-app-name.onrender.com`.

> ⚠️  Free Render instances spin down after 15 min of inactivity — the first request may take ~30 s.

---

### Option 2 — Railway (Free Tier)

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**.
2. Select your repository.
3. Railway auto-detects Python. Set the start command in settings:
   ```
   gunicorn app:app --bind 0.0.0.0:$PORT
   ```
4. Add environment variable: `PORT = 5000` (Railway sets `$PORT` automatically).
5. Deploy — Railway provides a public URL instantly.

---

### Option 3 — Hugging Face Spaces (Gradio / Streamlit alternative)

Hugging Face Spaces supports Flask via Docker:

1. Create a new Space → choose **Docker** as the SDK.
2. Add a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 7860
   ENV PORT=7860
   CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
   ```
3. Push your code (including `models/`) to the Space repo.
4. The Space will build and deploy automatically.

> Note: Hugging Face Spaces run on port **7860** by default.

---

## 🔧 Environment Variables

| Variable      | Default       | Description                    |
|---------------|---------------|--------------------------------|
| `PORT`        | `5000`        | Server port                    |
| `FLASK_DEBUG` | `true`        | Set to `false` in production   |
| `SECRET_KEY`  | `churn-dev-key-2024` | Flask secret key (change in prod!) |

---

## 📦 Model Files Reference

| File             | Where it comes from       | What it does                        |
|------------------|---------------------------|-------------------------------------|
| `churn_model.pkl`| Notebook cell — Step 10  | Trained XGBoost classifier          |
| `scaler.pkl`     | Notebook cell — Step 10  | StandardScaler for numeric features |

Both files are saved in the notebook with:
```python
joblib.dump(tuned_model, 'churn_model.pkl')
joblib.dump(scaler,      'scaler.pkl')
```

---

## 🛡️ Tech Stack

- **Backend:** Flask 3.0, Flask-CORS
- **ML:** XGBoost, scikit-learn, pandas, numpy
- **Frontend:** Bootstrap 5, Syne + DM Sans fonts
- **Server:** Gunicorn (production)
- **Model persistence:** joblib

---

## 📄 License

MIT
