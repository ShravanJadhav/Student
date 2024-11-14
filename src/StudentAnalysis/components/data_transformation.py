import sys
import os
from src.StudentAnalysis.exception import CustomException
from src.StudentAnalysis.logger import logging
from dataclasses import dataclass



@dataclass
class DataTranformationConfig:
    preprocessor_file_obj_path = os.join('artifacts','preprocessor.pkl')


class DataTransforamtion:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def get_data_transformation_obj(self):
        '''This is function for responsible for data transformation'''
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    

