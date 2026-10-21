"""Feature schema shared by training and inference."""

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"  # encoded as 1, the positive class
NEGATIVE_LABEL = "No"  # encoded as 0

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Milestone 1 only: the first working model uses two numeric inputs.
FIRST_MODEL_FEATURES = ["tenure", "MonthlyCharges"]
