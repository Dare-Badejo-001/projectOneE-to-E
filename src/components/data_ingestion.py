# here we read datasets from the different data sources and build a data pipeline 
# here we read the datasets 

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    split_ratio: float = 0.2
    data_source: str = "csv"


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.data_source = self.ingestion_config.data_source

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('raw data is saved')
            logging.info('train test split initiated')

            # Splitting the data into train and test sets
            # Here we are using the train_test_split function from sklearn to split the data
            train_set, test_set = train_test_split(df, test_size=self.ingestion_config.split_ratio, random_state=42)
            logging.info('train test split completed')
            # Saving the train and test sets to csv files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info('train data is saved')

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('test data is saved')

            logging.info("Data Ingestion method completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    logging.info('---- data ingestion started ----')
    data_ingestion = DataIngestion()
    train_data, test_data,_  = data_ingestion.initiate_data_ingestion()
    logging.info('---- data ingestion completed ----\n')
    logging.info('---- data transformation started ----')
    data_transformation = DataTransformation()
    train_arr, test_arr, _, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info('---- data transformation completed ----\n')
    logging.info('---- model trainer started ----')

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)
    logging.info('---- model trainer completed ----')
    