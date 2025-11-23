import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error


# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print("\nDataset Loaded Successfully.\n")
    print(df.head(), "\n")
    return df


# ----------------------------------------------------
# 2. TRAIN MODELS
# ----------------------------------------------------
def train_models(df):
    X = df[['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'Research']]
    y = df['Chance of Admit ']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42)

    # Train models
    lr.fit(X_train_scaled, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    pred_lr = lr.predict(X_test_scaled)
    pred_dt = dt.predict(X_test)
    pred_rf = rf.predict(X_test)

    # Print results
    print("📊 Model Performance:")
    evaluate("Linear Regression", y_test, pred_lr)
    evaluate("Decision Tree", y_test, pred_dt)
    evaluate("Random Forest", y_test, pred_rf)

    return lr, scaler


# ----------------------------------------------------
# 3. MODEL EVALUATION
# ----------------------------------------------------
def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{name}: R2 = {r2:.3f}, RMSE = {rmse:.3f}")


# ----------------------------------------------------
# 4. PREDICT FUNCTION
# ----------------------------------------------------
def predict_admission(model, scaler, gre, toefl, cgpa, sop, lor, research):
    features = pd.DataFrame([[gre, toefl, cgpa, sop, lor, research]],
                            columns=['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'Research'])
    features_scaled = scaler.transform(features)
    score = model.predict(features_scaled)[0] * 100
    return round(score, 2)


# ----------------------------------------------------
# 5. MAIN PROGRAM
# ----------------------------------------------------
if __name__ == "__main__":
    df = load_data("data/admission_data.csv")  
    model, scaler = train_models(df)

    example = predict_admission(
        model, scaler,
        gre=320, toefl=110, cgpa=8.5, sop=4, lor=4.5, research=1
    )

    print(f"\n🎓 Predicted Chance of Admit: {example}%\n")

