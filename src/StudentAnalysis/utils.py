import os  #for creatiing path and folder
import sys  # handling cutonException and logging
from src.StudentAnalysis.exception import CustomException
from src.StudentAnalysis.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

import pickle
import numpy as np

from  sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import r2_score

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db") 

def read_sql_data():
    logging.info("Reading SQL database started")

    try:
        mydb = pymysql.connect(
            host= host,
            user = user,
            password = password,
            db = db
        )

        logging.info("Connection Established", mydb)

        df = pd.read_sql_query('SELECT * FROM students',mydb)

        print(df)

        return df
    except Exception as ex:
        raise CustomException(ex,sys)


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models (x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for i in range (len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]
                
            gs = GridSearchCV(model,para,cv =3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

                #model.fit(x_train, y_train) # train Model

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
        return report
                    
    except Exception as e:
        raise CustomException(e,sys)