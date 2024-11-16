from src.StudentAnalysis.logger import logging
from src.StudentAnalysis.exception import CustomException
import sys
from src.StudentAnalysis.components.data_ingestion import DataIngestionConfig,DataIngestion
from src.StudentAnalysis.components.data_transformation import DataTranformationConfig, DataTransforamtion


if __name__=="__main__":
    logging.info("The execution has started")
    
    try:
        # dataIngestion
        #data_ingestion_Config = DataIngestionConfig()

        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.iniate_data_injestion()

        #dataTransformation

        #data_transformation_config = DataTranformationConfig()
        data_tranformation = DataTransforamtion()
        data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)