# Predict-Customer-Churn

What We're Predicting:

Each row is a telecom customer. We predict the probability they churn (cancel their service).
The dataset is synthetically generated from a deep learning model trained on the original IBM Telco churn dataset. Features include tenure, monthly charges, contract type, internet/phone services, and payment method.


| Feature Group | What we built |
|---|---|
| **Tenure** | Tenure groups, `is_new` / `is_mid` / `is_long` flags, log transform, inverse (`1/tenure+1`) |
| **Charges** | Log monthly, z-score, monthly vs mean ratio, high/low value flags |
| **Charge × Tenure** | `charges_per_tenure`, `tenure_x_monthly`, `log_t × log_m` |
| **Total Charges** | Deviation from expected (`monthly × tenure`), deviation %, `total_to_expected_ratio` |
| **Services** | Count of services, `charge_per_service`, streaming flag, no-services flag |
| **Contract** | Ordinal risk score (month-to-month=2, one year=1, two year=0), binary monthly flag |
| **Payment** | Auto-pay flag, electronic check flag |
| **Composite Risk** | Weighted score combining contract risk, auto-pay, new customer, service count |
| **Interactions** | `risk × services`, `risk × charge`, `senior × monthly`, `senior × risk` |



2. Encoding;

Label encoding used globally for Optuna tuning speed.
Target encoding applied inside each CV fold to prevent leakage:


te = TargetEncoder(cols=cat_cols, smoothing=10)
X_tr[cat_cols]  = te.fit_transform(X_tr[cat_cols], y_tr)  # fit on train fold only
X_val[cat_cols] = te.transform(X_val[cat_cols])           # apply to val


