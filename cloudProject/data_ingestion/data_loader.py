import numpy as np
import pandas as pd
class Data_Getter:
    def __init__(self, file_object, logger_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self,path):
        self.logger_object.log(self.file_object, "Entered the get_data method of the Data_Getter class")
        df=pd.read_csv(path)
        return df