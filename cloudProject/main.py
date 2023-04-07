from trainingmodel import trainModel
from trainingmodelH import trainModelHeart
from predictionmodel import predictModel1
from predictionmodel import predictModelH
import streamlit as st
st.title("Brain Stroke")
# here we define some of the front end elements of the web page like
# the font and background color, the padding and the text to be displayed
html_temp = """
<div style ="background-color:lightpink;padding:13px">
<h1 style ="color:black;text-align:center;">Brain Stroke</h1>
</div>
<br>
"""
st.markdown(html_temp, unsafe_allow_html = True)
col1,col2,col3=st.columns(3)
gender = col1.selectbox('Select your Gender',('Male', 'Female'))
age=col2.text_input("Enter your age")
hypertension=col3.selectbox('Select if hypertension',('0', '1'))
col4,col5,col6=st.columns(3)
ever_married=col4.selectbox('Are you ever married',('Yes', 'No'))
work_type=col5.selectbox('Select yourwork type',('Govt_jov', 'children','Never_worked','Private','Self-employed'))
Residence_type=col6.selectbox('Select residence area',('Rural', 'Urban'))
col7,col8,col9=st.columns(3)
avg_glucose_level=col7.text_input("Enter glucose level")
bmi=col8.text_input("Enter bmi")
smoking_status=col9.selectbox('Select your smoking status',('never smoked', 'formerly smoked','smokes','Unknown'))
col10,col11,col12=st.columns(3)
result =""
chestpain=col10.selectbox('Chest Pain type',('typical angina','atypical angina','non-anginal pain','asymptomatic'))
trtbps=col11.slider('Enter Resting blood pressure', 94.0, 200.0, 120.0)
chol=col12.slider("Enter cholostrol level",126.0,564.0,126.0)
col13,col14,col15=st.columns(3)
fbs=col13.selectbox("fasting blood sugar > 120 mg/dl",('1','0'))
exng=col14.selectbox("exercise induced angina",('1','0'))
thalachh=col15.slider("Enter maximum heart rate achieved",71.0,202.0,120.0)
col16,col17=st.columns(2)
oldpeak=col16.text_input("ST depression induced by exercise relative to rest")
slp1=col17.selectbox("the slope of the peak exercise ST segment",('upsloping','flat','downsloping'))
col18,col19=st.columns(2)
caa=col18.selectbox("number of major vessels",('0','1','2','3','4'))
restecg=col19.selectbox("rest ecg",('0','1','2'))

if st.button("Predict"):
    train_model=trainModel()
    train_model_heart=trainModelHeart()
    gender_map,ever_married_map,residence_type_map,smoking_status_map,work_type_map,best_model_name=train_model.trainingModels()
    predict_model=predictModel1()
    predict_modelh = predictModelH()
    train_model_heart.trainingModelsH()
    if gender=='Female':
        sex=0
    else:
        sex=1
    if chestpain=='asymptomatic':
        cp=4
    elif chestpain=='non-anginal pain':
        cp=3
    elif chestpain=='atypical angina':
        cp=2
    else:
        cp=1
    if slp1=='upsloping':
        slp=0
    elif slp1=='flat':
        slp=1
    else:
        slp=2
    heart_disease=predict_modelh.predictionFromModelH(age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa)
    result=predict_model.predictionFromModel(gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,gender_map,ever_married_map,residence_type_map,smoking_status_map,work_type_map,best_model_name)
    st.success('Suggested  is: {}'.format(result))