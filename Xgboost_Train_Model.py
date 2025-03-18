import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df=pd.read_csv("E:\\CSE\\4th Year 1st Semester\\( CSE 4180 ) Thesis_Project_(Part I)\\archive\\SMNI_CMI_TEST\\Data1.csv")
#print(df.head(20))

df1=df.iloc[:,3:13].values
#print(df1)

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
y = data_cleaned["matching condition"]



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=100)

# Train the XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


predictions = [round(value) for value in y_test_pred]
test_accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (test_accuracy*100.0))


#show precision    recall  f1-score   support
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
