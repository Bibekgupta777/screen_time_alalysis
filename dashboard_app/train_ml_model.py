# ==============================================
# ✅ TRAINING MODELS FOR STRESS & MENTAL HEALTH
# ==============================================

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned data
df = pd.read_csv("../data/cleaned_screen_time.csv")

# Encode categorical variables
le_gender = LabelEncoder()
le_location = LabelEncoder()

df["gender_encoded"] = le_gender.fit_transform(df["gender"])
df["location_encoded"] = le_location.fit_transform(df["location_type"])

# Features & Targets
features = ["age", "daily_screen_time_hours", "sleep_duration_hours", "gender_encoded", "location_encoded"]

X = df[features]
y_mental = df["mental_health_score"]
y_stress = df["stress_level"]

# Train Mental Health Model
X_train, X_test, y_train, y_test = train_test_split(X, y_mental, test_size=0.2, random_state=42)
mental_model = RandomForestRegressor(n_estimators=100, random_state=42)
mental_model.fit(X_train, y_train)
joblib.dump(mental_model, "../data/rf_mental_health_model.pkl")

# Train Stress Level Model
X_train, X_test, y_train, y_test = train_test_split(X, y_stress, test_size=0.2, random_state=42)
stress_model = RandomForestRegressor(n_estimators=100, random_state=42)
stress_model.fit(X_train, y_train)
joblib.dump(stress_model, "../data/rf_stress_level_model.pkl")

# Save label encoders too (for future predictions)
joblib.dump(le_gender, "../data/le_gender.pkl")
joblib.dump(le_location, "../data/le_location.pkl")

print("✅ Models trained & saved successfully!")
