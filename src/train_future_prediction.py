# src/train_future_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================================================
# CONFIG
# ==================================================
DATA_PATH = "Datasets/future_growth_dataset.csv"
MODEL_PATH = "models/future_growth_model.pkl"

RANDOM_SEED = 42

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(DATA_PATH)

X = df[["current_area"]]

y = df[[
    "future_30d",
    "future_60d",
    "future_90d"
]]

print("Total samples:", len(df))

# ==================================================
# SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_SEED
)

# ==================================================
# MODEL
# ==================================================
base_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

model = MultiOutputRegressor(base_model)

# ==================================================
# TRAIN
# ==================================================
model.fit(X_train, y_train)

# ==================================================
# PREDICT
# ==================================================
pred = model.predict(X_test)

# ==================================================
# METRICS
# ==================================================
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\n===== FUTURE MODEL RESULTS =====")
print("MAE  :", round(mae, 4))
print("RMSE :", round(rmse, 4))
print("R2   :", round(r2, 4))

# Per target R2
targets = ["30d", "60d", "90d"]

for i, name in enumerate(targets):
    score = r2_score(y_test.iloc[:, i], pred[:, i])
    print(f"R2 ({name}) :", round(score, 4))

# ==================================================
# SAVE
# ==================================================
joblib.dump(model, MODEL_PATH)

print("\nSaved model to:", MODEL_PATH)