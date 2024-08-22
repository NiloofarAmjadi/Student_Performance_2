import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the dataset
#url = 'https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/download'
df = pd.read_csv("C:/Users/amjad/Desktop/Data Science/ANA680 Machine Learning Deployment/week2/StudentPerformance/StudentsPerformance.csv")

# Data Exploration
print("First few rows of the dataset:")
print(df.head())

print("\nDataset summary:")
print(df.describe())

print("\nDataset info:")
print(df.info())

print("\nClass distribution:")
print(df['race/ethnicity'].value_counts())

# Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='race/ethnicity')
plt.title('Distribution of Race/Ethnicity')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='race/ethnicity', y='math score')
plt.title('Math Scores by Race/Ethnicity')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='race/ethnicity', y='reading score')
plt.title('Reading Scores by Race/Ethnicity')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='race/ethnicity', y='writing score')
plt.title('Writing Scores by Race/Ethnicity')
plt.show()

# Data Preprocessing
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluating the model
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

# Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the model
model_filename ='Random_Forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(pipeline, file)

print("\nModel saved as 'Random_Forest_model.pkl'")
