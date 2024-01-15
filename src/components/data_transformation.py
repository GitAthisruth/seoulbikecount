import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging


# def columndroping(data):
#     logging.info("column dropping function initialised")
#     data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
#     data.insert(
#         1, "weekday", data["Date"].dt.strftime("%w")
#     )  # here weekday starting from sunday as 0 .
#     data["weekday"] = pd.to_numeric(data["weekday"])
#     data.drop(["Date", "Dew point temperature(°C)"], axis=1, inplace=True)
#     return data


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation"""
        try:
            numerical_columns = [
                "weekday",
                "Hour",
                "Temperature(°C)",
                "Humidity(%)",
                "Wind speed (m/s)",
                "Visibility (10m)",
                "Solar Radiation (MJ/m2)",
                "Rainfall(mm)",
                "Snowfall (cm)",
            ]
            categorical_columns = ["Seasons", "Holiday", "Functioning Day"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [  # used to combine both pipelines together
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "Rented Bike Count"
            numerical_columns = [
                "weekday",
                "Hour",
                "Temperature(°C)",
                "Humidity(%)",
                "Wind speed (m/s)",
                "Visibility (10m)",
                "Solar Radiation (MJ/m2)",
                "Rainfall(mm)",
                "Snowfall (cm)",
            ]
            categorical_columns = ["Seasons", "Holiday", "Functioning Day"]
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            # input_feature_train_df = columndroping(train_df)
            # logging.info("Unwanted columns dropped successfully for train_data")

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            # input_feature_test_df = columndroping(test_df)
            # logging.info("Unwanted columns dropped successfully for test data")

            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
