import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


def columndroping(data):
    logging.info("column dropping function initialised")
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data.insert(
        1, "weekday", data["Date"].dt.strftime("%w")
    )  # here weekday starting from sunday as 0 .
    data["weekday"] = pd.to_numeric(data["weekday"])
    data.drop(["Date", "Dew point temperature(°C)"], axis=1, inplace=True)
    return data


@dataclass
class DataIngestionConfig:  # giving path to save train,test data and raw data.
    train_data_path: str = os.path.join(
        "artifacts", "train.csv"
    )  # train data will stored in this path.
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook\data\SeoulBikeData.csv")

            logging.info("Read the dataset as dataframe")
            columndroping(df)  # dropping unwanted columns
            logging.info("Unwanted columns dropped successfully from the raw data")
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=1)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of the data is completed")
            return (
                self.ingestion_config.train_data_path,  # used to grab the data path for next process
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)