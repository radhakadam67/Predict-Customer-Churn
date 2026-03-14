# Predict-Customer-Churn

What We're Predicting:

Each row is a telecom customer. We predict the probability they churn (cancel their service).
The dataset is synthetically generated from a deep learning model trained on the original IBM Telco churn dataset. Features include tenure, monthly charges, contract type, internet/phone services, and payment method.


Full Pipeline
1. Feature Engineering
Started with raw features and built meaningful signal on top:
Feature GroupWhat we builtTenureTenure groups, is_new / is_mid / is_long flags, log transform, inverse (1/tenure+1)ChargesLog monthly, z-score, monthly vs mean ratio, high/low value flagsCharge × Tenurecharges_per_tenure, tenure_x_monthly, log_t × log_mTotal chargesDeviation from expected (monthly × tenure), deviation %, total_to_expected_ratioServicesCount of services, charge_per_service, streaming flag, no-services flagContractOrdinal risk score (month-to-month=2, one year=1, two year=0), binary monthly flagPaymentAuto-pay flag, electronic check flagComposite riskWeighted score combining contract risk, auto-pay, new customer, service countInteractionsrisk × services, risk × charge, senior × monthly, senior × risk

