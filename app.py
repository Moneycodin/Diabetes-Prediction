import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
df=pd.read_csv('C:/Users/manme/OneDrive/Desktop/diabetes/diabetes.csv')
st.title('Diabetes Checkup in females')
st.subheader('Training Data')
st.write(df.describe())
st.subheader('Visualization')
st.bar_chart(df)
x=df.drop(['Outcome'], axis=1)
y=df.iloc[:,-1]
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=2)
def user_report():
    pregnancies=st.sidebar.slider('Pregnancies',0,17,3)
    glucose=st.sidebar.slider('Glucose',0,200,120)
    bp=st.sidebar.slider('Blood Pressure',0,122,70)
    skinthickness=st.sidebar.slider('Skin Thickness',0,100,20)
    insulin=st.sidebar.slider('Insulin',0,846,79)
    bmi=st.sidebar.slider('BMI',0,67,20)
    dpf=st.sidebar.slider('Diabetes Pedigree Function',0.0,2.4,0.47)
    age=st.sidebar.slider('Age',21,88,33)
    
    user_report={
       'Pregnancies':pregnancies,
       'Glucose':glucose,
       'BloodPressure':bp,
       'SkinThickness':skinthickness,
       'Insulin':insulin,
       'BMI':bmi,
       'DiabetesPedigreeFunction':dpf,
       'Age':age
    }  
    report_data=pd.DataFrame(user_report, index=[0])
    return report_data
user_data = user_report()
classifier=svm.SVC()
classifier.fit(x_train, y_train)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, classifier.predict(x_test))*100)+'%')

user_result=classifier.predict(user_data)
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'The person is not diabetic'
else:
  output = 'The person is diabetic'

st.write(output)