import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('../model/health_data.csv')

# Display initial info
print("Columns:", df.columns.tolist())
print("Null values:\n", df.isnull().sum())

# Encode categorical features
categorical_cols = ['Gender', 'Smoking', 'FamilyHistory']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save encoders for later

# Encode target variable
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# Save label encoder for target
with open('../model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save encoders for input columns
with open('../model/feature_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Split features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Check label distribution and mapping
print("\nClass Distribution:\n", y.value_counts())
print("\nLabel Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open('../model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved with encoders.")

