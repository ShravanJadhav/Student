import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.StudentAnalysis.exception import CustomException
from src.StudentAnalysis.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('articats','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test Data")

            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gardient Boosting" : GradientBoostingRegressor(),
                "Liner Regression" : LinearRegression(),
                "XGBRegressor" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(),
                "Adaboost Regressor" : AdaBoostRegressor()
            }

            params = {
                "Decision Tree" : {
                    'criterion' : ['squared_error','friedman_mse', 'absolute_error', 'poisson']
                   # 'splitter' : ['best','random'],
                   # 'max_features' : ['sqrt','log2']
                },

                "Random Forest" : {
                    #'criterion' : ['squared_error','friedman_mse', 'absolute_error', 'poisson'],
                    #'max_features' : ['sqrt','log2', 'none'],
                    'n_estimators' : [8,16,32,64,128,256]
                },

                "gardient Boosting" : {
                    #'loss' : ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate' : [.1,.01,.05,.001],
                    'subsample' : [0.6,0.7,0.75,0.9,0.85,0.9],
                    #'crierion' : ['squared_error','friedman_mse'],
                    #'max_features' : ['auto','sqrt', 'log2'],
                    'n_estimators' : [8,16,32,64,128,256]
                },

                'Liner Regression' : {},

                'XGBRegressor' : {
                    'learning_rate' : [.1,.01,.05,.001],
                    'n_estimators' : [8,16,32,64,128,256]
                },

                "catBoosting Regressor" :{
                    'depth' : [6,8,10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations' : [30,50,100]
                },

                "AdaBoost Regressor" : {
                    'learning_rate' : [.1,.01,0.5,.001],
                   # 'loss' : [0.01, 0.05, 0.1],
                    'n_estimators' : [8,16,32,64,128,226]

                }
            }

        except Exception as e:
            raise CustomException(e,sys)