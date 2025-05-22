import streamlit as st
import pickle
import sklearn


st.header("Titanic Survival Prediction")
st.subheader("Predicting Survival on the Titanic")
st.image("titanic.jpeg")
st.text('''The Titanic was a British luxury ship that sank on its maiden voyage from Southampton to New York in            April 1912 after hitting an iceberg. Over 1,500 of the 2,200+ people on board died, making it one of the          deadliest maritime disasters. The tragedy led to major improvements in ship safety regulations.''')

model=pickle.load(open('model.pkl','rb'))
l_sex=pickle.load(open('l_sex.pkl','rb'))
l_emb=pickle.load(open('l_emb.pkl','rb'))

pclass=st.number_input("Passenger Class")
#pclass=st.radio("select passsenger class",(1,2,3))------> selection option instead of passenger class input
sex=st.text_input("enter sex:[male,female]")
age=st.number_input("Age")
sibsp=st.number_input("Number of Siblings/Spouses Abroad")
parch=st.number_input("Number of parents/children Abroad")
fare=st.number_input("Fare")
embarked=st.text_input("Embarked:[S,Q,C]")
if st.button("Predict"):
    sex_l=l_sex.transform([sex])[0]
    embarked_l=l_emb.transform([embarked])[0]
    predict=model.predict([[pclass,sex_l,age,sibsp,parch,fare,embarked_l]])
    if predict==1:
       st.success('Survived')
    else:
       st.error('Did not survive')
      #st.warning 
