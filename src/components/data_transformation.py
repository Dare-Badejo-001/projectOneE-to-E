# the main purpose here is to do data cleaning and feature engineering

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    target_encoder_file_path = os.path.join('artifacts', 'target_encoder.pkl')
    target_scaler_file_path = os.path.join('artifacts', 'target_scaler.pkl')
    model_file_path = os.path.join('artifacts', 'model.pkl')
    model_config_file_path = os.path.join('artifacts', 'model_config.yaml')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessor = None
        self.target_encoder = None
        self.target_scaler = None

    def get_data_transformer_object(self):
        '''
        This function is responsible for transforming the data
        '''

        try:
            numerical_features = ['reading_score','writing_score']
            categorical_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())

            ])

            logging.info("Numerical columns standard scaling completed successfully")
            logging.info(f"Numerical features: {numerical_features}")


            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info("Categorical columns encoding completed successfully")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            logging.info("Preprocessor pipeline created with numerical and categorical transformations successfully configured.")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function is responsible for transforming the data
        '''
        try:
             # read the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test completed successfully")
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")


            # get the preprocessor object
            logging.info("Getting the preprocessor object")
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name  = "math_score"
                
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info("Train and test dataframes split into input features and target feature successfully")
            
            
            logging.info("Applying preprocessing object on training and testing dataframes")
            preprocessing_obj.fit(input_features_train_df)
            input_features_train_df = preprocessing_obj.transform(input_features_train_df)
            input_features_test_df = preprocessing_obj.transform(input_features_test_df)
            logging.info("Preprocessing object applied on training and testing dataframes successfully")
            
            train_arr = np.c_[input_features_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_df, np.array(target_feature_test_df)]
            logging.info("Train and test arrays created successfully")
            
            logging.info("Saving preprocessing object")

            save_object(  
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                    )
            logging.info("Preprocessing object saved successfully")

            return (
                    train_arr,
                    test_arr,
                    preprocessing_obj, 
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
     
        except Exception as e:
            raise CustomException(e, sys)
 
