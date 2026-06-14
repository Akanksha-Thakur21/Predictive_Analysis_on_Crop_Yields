import pandas as pd
df = pd.read_csv('Crop_production.csv')
print(df.info()) 
print(df.isnull().sum())

# Step 1: Data Cleaning
df = df.drop("Unnamed: 0", axis=1)
df.fillna(df.mean(numeric_only=True), inplace=True)

print("Data cleaned successfully!")

# Step 2: Statistical Analysis

print("\nDataset Statistics:")
print(df.describe())

# Step 3: Visualization

import matplotlib.pyplot as plt

avg_yield = df.groupby("Crop")["Yield_ton_per_hec"].mean()

plt.figure(figsize=(12,6))
avg_yield.sort_values().plot(kind="bar")

plt.title("Average Crop Yield")
plt.xlabel("Crop")
plt.ylabel("Yield (ton/hectare)")

plt.tight_layout()
plt.show()

# Step 4: Convert Text Data into Numbers

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["State_Name"] = le.fit_transform(df["State_Name"])
df["Crop_Type"] = le.fit_transform(df["Crop_Type"])
df["Crop"] = le.fit_transform(df["Crop"])
print(df.head())

# Step 5: Define Features and Target

X = df.drop("Yield_ton_per_hec", axis=1)
y = df["Yield_ton_per_hec"]

# Step 6: Split Dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Step 7: Train Linear Regression Model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")

# Step 8: Predict Crop Yield

predictions = model.predict(X_test)

print(predictions[:5])

# Step 9: Evaluate Model

from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)

print("R2 Score:", score)

# Step 10: Save Predictions

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

results.to_csv("output/predictions.csv", index=False)

print("Predictions saved successfully!")