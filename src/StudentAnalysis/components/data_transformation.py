import sys
import os
from src.StudentAnalysis.exception import CustomException
from src.StudentAnalysis.logger import logging
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



@dataclass
class DataTranformationConfig:
    preprocessor_file_obj_path = os.join('artifacts','preprocessor.pkl')


class DataTransforamtion:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def get_data_transformation_obj(self):
        '''This is function for responsible for data transformation'''
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            #Handling Missing values

            num_pipeline = Pipeline(steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())
            ])

            cat_pipeline = Pipeline(steps = [
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoding', OneHotEncoder())
                ('scalar',StandardScaler(with_mean=False))
            ])

            logging.info("Categorical Columns : {categorical_columns}")
            logging.info("numerical Columns : {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    

