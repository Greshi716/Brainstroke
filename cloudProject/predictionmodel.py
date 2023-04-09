from application_logging.logger import App_Logger
from file_operations.file_methods import File_Operation
import pickle
class predictModel1:
    def __init__(self):
        self.file_object=open("/app/brainstroke/cloudProject/Prediction_Logs/Prediction_Log.txt",'a+')
        self.log_writer=App_Logger()
    def predictionFromModel(self,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,gender_map,ever_married_map,residence_type_map,smoking_status_map,work_type_map,model_name):
        gender=gender_map.get(gender)
        ever_married=ever_married_map.get(ever_married)
        Residence_type=residence_type_map.get(Residence_type)
        smoking_status=smoking_status_map.get(smoking_status)
        work_type=work_type_map.get(work_type)
        file_op = File_Operation(self.file_object, self.log_writer)
        load_model = file_op.load_model(model_name)
        lst=load_model.predict_proba([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])[0]
        lst=lst.tolist()
        index=max(lst)
        prediction = lst.index(index)
        print(prediction)
        if prediction==0:
            return "No stroke"
        else:
            return "Chances of stroke"

class predictModelH:
    def __init__(self):
        self.file_object=open("/app/brainstroke/cloudProject/Prediction_Logs/Prediction_Log.txt",'a+')
        self.log_writer=App_Logger()
    def predictionFromModelH(self,age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa):
        file_op = File_Operation(self.file_object, self.log_writer)
        load_model =pickle.load(open('/app/brainstroke/cloudProject/LogisticRegression.pkl', 'rb'))
        lst=load_model.predict_proba([[age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa]])[0]
        lst=lst.tolist()
        index=max(lst)
        prediction = lst.index(index)
        print(prediction)
        if prediction==0:
            return 0
        else:
            return 1
