from src.StudentAnalysis.logger import logging
from src.StudentAnalysis.exception import CustomException
import sys
from src.StudentAnalysis.components.data_ingestion import DataIngestionConfig,DataIngestion



if __name__=="__main__":
    logging.info("The execution has started")
    
    try:
        
        #data_ingestion_Config = DataIngestionConfig()

        data_ingestion = DataIngestion()
        data_ingestion.iniate_data_injestion()


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)