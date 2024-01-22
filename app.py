from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)


app=application

##Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Hour = request.form.get("Hour"),
            Temperature = float(request.form.get("Temperature(°C)")),
            Humidity = float(request.form.get("Humidity(%)")),
            Wind_speed = float(request.form.get("Wind speed (m/s)")),
            Visibility = float(request.form.get("Visibility (10m)")),
            Solar_Radiation = float(request.form.get("Solar Radiation (MJ/m2)")),
            Rainfall = float(request.form.get("Rainfall(mm)")),
            Snowfall = float(request.form.get("Snowfall (cm)")),
            weekday = request.form.get("weekday"),
            Seasons = request.form.get("Seasons"),
            Holiday = request.form.get("Holiday"),
            Functioning_Day = request.form.get("Functioning Day")

        )

        pred_df=data.get_data_as_data_frame()
        # print("pred_df printed",pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        # print("results",results)
        return render_template('home.html',results=abs(int(results[0])))
    
if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)#remove debug=True during deployment

