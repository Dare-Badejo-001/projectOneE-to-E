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
import numpy as np
from sklearn.model_selection import GridSearchCV

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

        self.params = {
        "DecisionTreeRegressor": {
            "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            "splitter": ["best", "random"],
            "max_depth": [None] + list(np.linspace(10, 30, 3, dtype=int)),    
            "min_samples_split": list(np.linspace(2, 10, 3, dtype=int)),
            "min_samples_leaf": list(np.linspace(1, 5, 3, dtype=int)),
        },
        "KNeighborsRegressor": {
            "n_neighbors": list(np.linspace(3, 9, 4, dtype=int)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(np.linspace(10, 30, 3, dtype=int)),
        },
        "RandomForestRegressor": {
            "n_estimators": list(np.logspace(1, 2.5, 3, dtype=int)),  # ~10, 32, 100
            "criterion": ["squared_error", "absolute_error", "poisson"],
            "max_depth": [None] + list(np.linspace(10, 30, 3, dtype=int)),
            "min_samples_split": list(np.linspace(2, 10, 3, dtype=int)),
            "min_samples_leaf": list(np.linspace(1, 4, 3, dtype=int)),
        },
        "GradientBoostingRegressor": {
            "n_estimators": list(np.logspace(2, 2.3, 2, dtype=int)),  # 100, 200
            "learning_rate": np.logspace(-2, -0.7, 3),  # 0.01, ~0.05, 0.2
            "max_depth": list(np.linspace(3, 7, 3, dtype=int)),
            "min_samples_split": list(np.linspace(2, 5, 2, dtype=int)),
            "min_samples_leaf": list(np.linspace(1, 2, 2, dtype=int)),
        },
        "XGBRegressor": {
            "n_estimators": list(np.logspace(2, 2.3, 2, dtype=int)),
            "learning_rate": np.logspace(-2, -0.7, 3),
            "max_depth": list(np.linspace(3, 7, 3, dtype=int)),
            "min_samples_split": list(np.linspace(2, 5, 2, dtype=int)),
            "min_samples_leaf": list(np.linspace(1, 2, 2, dtype=int)),
        },
        "CatBoostRegressor": {
            "iterations": list(np.logspace(2, 2.3, 2, dtype=int)),
            "learning_rate": np.logspace(-2, -0.7, 3),
            "depth": list(np.linspace(3, 7, 3, dtype=int)),
            "l2_leaf_reg": list(np.logspace(0, 1, 3, dtype=int)),  # 1, 3, 10
        },
        "AdaBoostRegressor": {
            "n_estimators": list(np.logspace(1.7, 2, 2, dtype=int)),  # 50, 100
            "learning_rate": np.logspace(-2, -0.7, 3),
        },
        "LinearRegression": {
        # No hyperparameters to tune for base LinearRegression
    },
    }
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        model_report = {}
        for model_name, model in self.models.items():
            try:
                logging.info(f"Training {model_name} with hyperparameter tuning and GridSearchCV")
                param_grid = self.params[model_name]
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring="r2",
                    cv=3,
                    verbose=1,
                    n_jobs=-1,
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                logging.info(f"Best score for {model_name}: {grid_search.best_score_}")
                logging.info("Training the best model on the entire training set")
                best_model.fit(X_train, y_train)
                logging.info("Predicting on the test set")
                y_test_pred = best_model.predict(X_test)
                logging.info("Calculating R2 score")
                r2_square = r2_score(y_test, y_test_pred)
                logging.info(f"R2 Score test data: {r2_square * 100:.2f}%")
                logging.info("Model trained successfully")
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