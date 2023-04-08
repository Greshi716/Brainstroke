from data_ingestion.data_loader import Data_Getter
from application_logging.logger import App_Logger
from file_operations import file_methods
# from data_visualizing.datavisualizer import datavisualizer
from data_preprocessing import preproessing
from sklearn.model_selection import train_test_split
from best_model_finder import tuner
import pickle

class trainModel:
    def __init__(self):
        self.log_writer = App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModels(self):
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            data_getter = Data_Getter(self.file_object, self.log_writer)
            data=data_getter.get_data("C:/Users/Asus/Downloads/brain_stroke.csv")
            preprocessor = preproessing.Preprocessor(self.file_object, self.log_writer)
            data,gender_map,ever_married_map,residence_type_map,smoking_status_map,work_type_map=preprocessor.labelencoding(data)
            data['work_type'] = data['work_type'].fillna(data['work_type'] == 0)
            data['age'] = data['age'].astype(int)
            data['work_type'] = data['work_type'].astype(int)
            data['avg_glucose_level'] = data['avg_glucose_level'].astype(int)
            data['bmi'] = data['bmi'].astype(int)
            # data['work_type'].isnull().sum
            X, Y = preprocessor.seperate_features_target(data, label_column_name='stroke')
            X = preprocessor.features_selection(X)
            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=0)
            xtrainnew,ytrainnew=preprocessor.performsmote(xtrain,ytrain)
            model_finder=tuner.Model_finder(self.file_object,self.log_writer)
            best_model_name,best_model=model_finder.get_best_model(xtrainnew,ytrainnew,xtest,ytest)
            file_op = file_methods.File_Operation(self.file_object, self.log_writer)
            save_model = file_op.save_model(best_model, best_model_name)
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()
            return gender_map,ever_married_map,residence_type_map,smoking_status_map,work_type_map,best_model_name
        except Exception:
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
