import numpy as np
import pandas as pd
df=pd.read_csv("E:\\CSE\\4th Year 1st Semester\\( CSE 4180 ) Thesis_Project_(Part I)\\archive\\SMNI_CMI_TEST\\Data1.csv")
print(df.head(10))
df.shape
df.isnull().sum()
print(df.isnull().sum())



df1=df.drop(['sample num','subject identifier','matching condition','name'],axis=1)
#print(df1)
print(df1.head())
x=df1.iloc[:,0:5]
print(x.head())
y=df1.iloc[:,5:6]
print(y.head())
#y=df['sample num']
#print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.5,random_state=30)
xtrain.shape
print("Xtrain :")
print(xtrain)

print("Ytrain :")
ytrain.shape
print(ytrain)


print("Xtest : ")
xtest.shape
print(xtest)

print("Ytest :")
ytest.shape
print(ytest)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
xtrain['column_name'] = label_encoder.fit_transform(xtrain['column_name'])




from sklearn.svm import SVC
SV=SVC()
SV.fit(xtrain,ytrain)
model=SVC()
SV.score(xtest,ytest)
print(SV.score(xtest,ytest))