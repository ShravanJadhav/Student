import os  #for creatiing path and folder
import sys  # handling cutonException and logging
from src.StudentAnalysis.exception import CustomException
from src.StudentAnalysis.logger import logging
import pandas as pd

from dataclasses import dataclass  #input parameters initiate

from src.StudentAnalysis.utils import read_sql_data

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    row_data_path:str = os.path.join('artifacts','row.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
    
    def iniate_data_injestion(self):
        try:
            #Reading Data from database
            #df = read_sql_data()
            df = pd.read_csv('notebook/data/row.csv')
            logging.info("Reading data from mysql")

            #creating artifact folder and file if its dosenot exits
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.row_data_path, index=False,header=True)

            train_set, test_set = train_test_split(df,test_size = 0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Data Ingestion Complated")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)
