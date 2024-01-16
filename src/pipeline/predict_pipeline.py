import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts\\model.pkl"
            preprocessor_path="artifacts\\preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            # print("preprocessor",preprocessor)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            # print("preds",data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        


class CustomData:
    def __init__(self,
                Hour,
                Temperature,
                Humidity,
                Wind_speed,
                Visibility,
                Solar_Radiation,
                Rainfall,
                Snowfall,
                weekday,
                Seasons,
                Holiday,
                Functioning_Day
                                    ):
        
        self.Hour = Hour
        self.Temperature = Temperature
        self.Humidity = Humidity
        self.Wind_speed = Wind_speed
        self.Visibility = Visibility
        self.Solar_Radiation = Solar_Radiation
        self.Rainfall = Rainfall
        self.Snowfall = Snowfall
        self.weekday = weekday
        self.Seasons = Seasons
        self.Holiday = Holiday
        self.Functioning_Day = Functioning_Day

    def get_data_as_data_frame(self):#The values giving in the webpage will assign to this variables. 
        try:
            custom_data_input_dict={
            'weekday':[self.weekday],
            'Hour':[self.Hour],
            'Temperature(°C)':[self.Temperature],
            'Humidity(%)':[self.Humidity],
            'Wind speed (m/s)':[self.Wind_speed],
            'Visibility (10m)':[self.Visibility],
            'Solar Radiation (MJ/m2)':[self.Solar_Radiation],
            'Rainfall(mm)':[self.Rainfall],
            'Snowfall (cm)':[self.Snowfall],
            'Seasons':[self.Seasons],
            'Holiday':[self.Holiday],
            'Functioning Day':[self.Functioning_Day]
            }
            # print(custom_data_input_dict)
            # print(pd.DataFrame(custom_data_input_dict))
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)


