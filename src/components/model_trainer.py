# here we will train different models, 
import os 
import sys 

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "AdaBoostRegressor": AdaBoostRegressor(),
        }
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        model_report = {}
        for model_name, model in self.models.items():
            try:
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                r2_square = r2_score(y_test, y_test_pred)
                logging.info("Model trained successfully")
                logging.info(f"R2 Score test data: {r2_square * 100:.2f}%")
                model_report[model_name] = r2_square
            except Exception as e:
                logging.error(f"Error occurred while training {model_name}: {e}")
                continue
        return model_report
    

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Training models")

            model_report: dict = self.evaluate_models(X_train, y_train, X_test, y_test)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = self.models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")
            logging.info("Best model found:") 
            logging.info(f"Model choice is {best_model_name} with score: {best_model_score * 100:.2f}%")
            logging.info("Saving the best model")   
            logging.info("Saving the best model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved successfully")
            
        except Exception as e:
            raise CustomException(e, sys)   