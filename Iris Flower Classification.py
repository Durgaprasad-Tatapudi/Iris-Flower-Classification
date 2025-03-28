# Import necessary libraries
import kagglehub
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("arshid/iris-flower-dataset")
print("Path to dataset files:", dataset_path)

# Step 2: List available files
available_files = os.listdir(dataset_path)
print("Available files in dataset:", available_files)

# Step 3: Correct dataset filename
file_name = "IRIS.csv"  # Ensure correct filename
data_path = os.path.join(dataset_path, file_name)

# Step 4: Load dataset
def load_data(file_path):
    """Loads a dataset file into a Pandas DataFrame."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()  # Convert column names to lowercase
    return df

df = load_data(data_path)
print("\nDataset loaded successfully.")
print(df.head())

# Step 5: Check column names
print("\nColumns in dataset:", df.columns)

# Step 6: Handle missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 7: Encode categorical target variable (species)
label_encoder = LabelEncoder()
df["species"] = label_encoder.fit_transform(df["species"])

# Step 8: Select features and target
X = df.drop(columns=["species"])
y = df["species"]

# Step 9: Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Normalize numerical data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Step 11: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 12: Make predictions
y_pred = model.predict(X_val)

# Step 13: Evaluate model
accuracy = accuracy_score(y_val, y_pred)
classification_rep = classification_report(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Step 14: Save predictions
submission = pd.DataFrame({"Sample_ID": df.index[:len(y_pred)], "Predicted_Species": label_encoder.inverse_transform(y_pred)})
submission_path = os.path.join(dataset_path, "iris_predictions.csv")
submission.to_csv(submission_path, index=False)

print(f"\nPredictions saved at: {submission_path}")

# Step 15: Visualization

# 1. Pairplot for Feature Distribution
sns.pairplot(df, hue="species", palette="husl")
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# 3. Feature Importance Bar Chart
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
feature_importance.nlargest(4).plot(kind="barh", color="teal")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")
plt.show()
