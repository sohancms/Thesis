import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
control_data_path = "E:\\CSE\\4th Year 2nd Semester\\Thesis\\archive2\\Pivoted_EEG_Alcohol_Data\\EEG_data_control_pivoted.csv"
experimental_data_path = "E:\\CSE\\4th Year 2nd Semester\\Thesis\\archive2\\Pivoted_EEG_Alcohol_Data\\EEG_data_pivoted.csv"

# Read the datasets
control_data = pd.read_csv(control_data_path)
experimental_data = pd.read_csv(experimental_data_path)

# Add labels to each dataset
control_data['label'] = 0  # Label 0 for the control group
experimental_data['label'] = 1  # Label 1 for the experimental group

# Combine the datasets
data = pd.concat([control_data, experimental_data], axis=0)

# Separate features (X) and labels (y)
X = data.drop(columns=['label', 'time'])  
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')  # Disable label encoding warnings
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy*100.0))
print("Classification Report:")
print(classification_report(y_test, y_pred))
