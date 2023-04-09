from data_ingestion.data_loader import Data_Getter
from application_logging.logger import App_Logger
from file_operations import file_methods
from data_preprocessing import preproessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle
class trainModelHeart:
    def __init__(self):
        self.log_writer = App_Logger()
        self.file_object = open("/app/brainstroke/cloudProject/Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModelsH(self):
        data_getter = Data_Getter(self.file_object, self.log_writer)
        data = data_getter.get_data('/app/brainstroke/cloudProject/heart attack.csv')
        data.drop(columns=['thall'],inplace=True)
        preprocessor = preproessing.Preprocessor(self.file_object, self.log_writer)
        X, Y = preprocessor.seperate_features_target(data, label_column_name='output')
        X = preprocessor.features_selection(X)
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=70)
        xtrainnew, ytrainnew = preprocessor.performsmote(xtrain, ytrain)
        lr = LogisticRegression()
        lr.fit(xtrainnew,ytrainnew)
        preds = lr.predict(xtest)
        accuracy_score(ytest, preds)
        print(accuracy_score(ytest, preds))
        print(classification_report(ytest, preds))
        file_op = file_methods.File_Operation(self.file_object, self.log_writer)
        # save_model = file_op.save_model('LogisticRegression','LogisticRegression')
        pickle.dump(lr, open('LogisticRegression.pkl', 'wb'))
        self.log_writer.log(self.file_object, 'Successful End of Training')
        self.file_object.close()

