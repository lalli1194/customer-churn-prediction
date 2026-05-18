"""
╔══════════════════════════════════════════════════════════════════╗
║        TELCO CUSTOMER CHURN PREDICTION — FLASK BACKEND           ║
║  Author  : Your Name                                             ║
║  Purpose : Serve the trained XGBoost churn model via Flask       ║
╚══════════════════════════════════════════════════════════════════╝

HOW THE APP WORKS
─────────────────
  GET  /            → Renders the prediction form (index.html)
  POST /predict     → Accepts form data OR JSON, returns prediction
  GET  /health      → Returns API health status (useful for deploy checks)

FILE LAYOUT
───────────
  app.py                ← YOU ARE HERE
  models/
    churn_model.pkl     ← Trained XGBoost model (from notebook step 10)
    scaler.pkl          ← Fitted StandardScaler  (from notebook step 10)
  templates/index.html  ← Bootstrap + custom UI
  static/style.css      ← Custom styles
  requirements.txt
  README.md
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import traceback

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════════════════════════
#  APP INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (needed for API consumers / Postman)

# ── Secret key (change in production) ─────────────────────────────────────────
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'churn-dev-key-2024')

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
#  We load once at startup so every request is fast.
#  Place churn_model.pkl and scaler.pkl inside the  models/  folder.
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

def load_artifacts():
    """Load model and scaler from disk.  Returns (model, scaler) or raises."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Copy churn_model.pkl from your notebook into the models/ folder."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}.\n"
            "Copy scaler.pkl from your notebook into the models/ folder."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    MODEL, SCALER = load_artifacts()
    print("✅  Model and scaler loaded successfully.")
except FileNotFoundError as e:
    MODEL, SCALER = None, None
    print(f"⚠️  WARNING: {e}")
    print("    The app will start, but /predict will return an error until "
          "the model files are present.")

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
#  This MUST match what was done during training in the notebook.
#  Any mismatch → wrong predictions.
# ═══════════════════════════════════════════════════════════════════════════════

# Columns that were One-Hot Encoded during training (drop_first=True)
# These are the ORIGINAL categorical columns with 3+ categories.
MULTI_CAT_COLS = ['InternetService', 'Contract', 'PaymentMethod']

# Binary columns that were Label-Encoded (Yes/No, Male/Female → 1/0)
BINARY_MAP = {
    'Yes': 1, 'No': 0,
    'Male': 1, 'Female': 0,
}

# Numeric columns that were scaled with StandardScaler
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Exact column order that the trained model expects
# (derived by running:  print(X.columns.tolist())  in the notebook after encoding)
EXPECTED_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'InternetService_Fiber optic', 'InternetService_No',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check',
]


def preprocess(raw: dict) -> pd.DataFrame:
    """
    Apply the same preprocessing pipeline used during training.

    Steps
    ─────
    1. Cast numeric fields
    2. Label-encode binary categoricals
    3. One-Hot-Encode multi-category fields
    4. Align columns to training order (fill missing OHE columns with 0)
    5. Scale numeric columns
    6. Return a single-row DataFrame ready for model.predict()
    """
    df = pd.DataFrame([raw])

    # ── Step 1 · Numeric casting ─────────────────────────────────────────────
    df['tenure']         = pd.to_numeric(df['tenure'],         errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['TotalCharges']   = pd.to_numeric(df['TotalCharges'],   errors='coerce').fillna(0)
    df['SeniorCitizen']  = pd.to_numeric(df['SeniorCitizen'],  errors='coerce').fillna(0).astype(int)

    # ── Step 2 · Label-encode binary columns ─────────────────────────────────
    binary_cols = [c for c in df.columns
                   if c not in MULTI_CAT_COLS + NUMERIC_COLS + ['SeniorCitizen']
                   and df[c].dtype == object]

    for col in binary_cols:
        df[col] = df[col].map(BINARY_MAP).fillna(0).astype(int)

    # ── Step 3 · One-Hot Encode multi-category columns ────────────────────────
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)

    # ── Step 4 · Align to training columns (add missing OHE dummies as 0) ────
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[EXPECTED_COLUMNS]          # enforce exact column order

    # ── Step 5 · Scale numeric features ─────────────────────────────────────
    df[NUMERIC_COLS] = SCALER.transform(df[NUMERIC_COLS])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER : BUILD HUMAN-FRIENDLY RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def build_response(prob_churn: float) -> dict:
    """Convert raw churn probability into a structured result dict."""
    will_churn = prob_churn >= 0.50

    if prob_churn >= 0.70:
        risk_level = "High"
        risk_color = "danger"
        message    = ("⚠️ This customer is very likely to churn. "
                      "Immediate retention action is strongly recommended.")
    elif prob_churn >= 0.40:
        risk_level = "Medium"
        risk_color = "warning"
        message    = ("🔔 This customer shows moderate churn risk. "
                      "Consider a proactive outreach or a loyalty offer.")
    else:
        risk_level = "Low"
        risk_color = "success"
        message    = ("✅ This customer is likely to stay. "
                      "Keep up the great service!")

    return {
        "prediction"        : "Churn"    if will_churn else "No Churn",
        "churn_probability" : round(float(prob_churn) * 100, 2),
        "stay_probability"  : round((1 - float(prob_churn)) * 100, 2),
        "risk_level"        : risk_level,
        "risk_color"        : risk_color,
        "message"           : message,
        "will_churn"        : bool(will_churn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def home():
    """Render the prediction form."""
    model_ready = MODEL is not None
    return render_template('index.html', model_ready=model_ready)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accept customer data (form OR JSON) → return churn prediction.

    Form POST  → re-renders index.html with result embedded
    JSON POST  → returns JSON  (for API / Postman usage)

    Sample JSON input:
    {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 2,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 79.85,
      "TotalCharges": 159.70
    }
    """
    # ── Guard: model not loaded ───────────────────────────────────────────────
    if MODEL is None or SCALER is None:
        error_msg = ("Model files not found. "
                     "Place churn_model.pkl and scaler.pkl in the models/ folder.")
        if request.is_json:
            return jsonify({"error": error_msg}), 503
        return render_template('index.html', error=error_msg, model_ready=False)

    # ── Determine request source ──────────────────────────────────────────────
    is_api = request.is_json or request.headers.get('Accept') == 'application/json'

    try:
        # ── Parse input ───────────────────────────────────────────────────────
        if request.is_json:
            raw = request.get_json(force=True)
        else:
            raw = request.form.to_dict()

        if not raw:
            raise ValueError("No input data received.")

        # ── Preprocess ────────────────────────────────────────────────────────
        processed = preprocess(raw)

        # ── Predict ───────────────────────────────────────────────────────────
        prob_array  = MODEL.predict_proba(processed)[0]   # [prob_no_churn, prob_churn]
        prob_churn  = prob_array[1]

        result = build_response(prob_churn)

        # ── Return ────────────────────────────────────────────────────────────
        if is_api:
            return jsonify({"status": "success", "result": result}), 200
        return render_template('index.html',
                               result=result,
                               form_data=raw,
                               model_ready=True)

    except ValueError as ve:
        err = str(ve)
        if is_api:
            return jsonify({"status": "error", "message": err}), 400
        return render_template('index.html', error=err, model_ready=True)

    except Exception:
        err = traceback.format_exc()
        print(f"[ERROR] /predict:\n{err}")
        friendly = "An internal error occurred. Please check server logs."
        if is_api:
            return jsonify({"status": "error", "message": friendly}), 500
        return render_template('index.html', error=friendly, model_ready=True)


@app.route('/health', methods=['GET'])
def health():
    """API health check — useful for Render / Railway uptime monitoring."""
    return jsonify({
        "status"     : "ok",
        "model_loaded": MODEL is not None,
        "version"    : "1.0.0",
    }), 200


@app.route('/api/sample', methods=['GET'])
def sample_input():
    """Return a sample JSON payload — handy for API consumers."""
    sample = {
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
        "TotalCharges"    : 159.70,
    }
    return jsonify({"sample_input": sample}), 200


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    print(f"\n🚀  Starting Churn Prediction API on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
