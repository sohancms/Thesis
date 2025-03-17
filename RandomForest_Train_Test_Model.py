import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"E:\CSE\4th Year 1st Semester\( CSE 4180 ) Thesis_Project_(Part I)\archive\SMNI_CMI_TEST\Data1.csv")

# Drop unnecessary columns
data_cleaned = df.drop(columns=["Unnamed: 0", "name"])

# Encode categorical variables
label_encoders = {}
categorical_columns = ["sensor position", "subject identifier", "matching condition"]

for col in categorical_columns:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])  
    label_encoders[col] = le

# Define features (X) and target (y)
X = data_cleaned.drop(columns=["matching condition"])
y = data_cleaned["matching condition"].astype(int)  # Ensure it's numeric

# Reset indices (Optional)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_test_pred = model.predict(X_test)

# Compute accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2%}")
