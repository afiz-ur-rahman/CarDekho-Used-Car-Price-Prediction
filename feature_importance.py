import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model and feature names
model = joblib.load("car_price_model.pkl")
feature_names = joblib.load("feature_columns.pkl")

# Get feature importance
importances = model.feature_importances_

# Create DataFrame
fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort by importance
fi_df = fi_df.sort_values(by="Importance", ascending=False).head(15)

# Print top features
print("\nTop 15 Important Features:")
print(fi_df)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(fi_df["Feature"], fi_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances for Car Price Prediction")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
