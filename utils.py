from pandas import Timestamp, DateOffset
import pickle
import os

class Utils():
    def __init__(self, company:str,):
        last_date = Timestamp('2024-10-01 00:00:00') # taken from notebook TODO: change to dynamic
        self.get_forecast_dates = lambda steps: [last_date + DateOffset(months=i) for i in range(1, steps+1)]

        self.models = {}
        for file in os.listdir(f'./config/{company}/ml'):
            if file.endswith('.pkl'):
                with open(os.path.join(f'./config/{company}/ml', file), 'rb') as f:
                    self.models[os.path.splitext(file)[0]] = pickle.load(f)

    def predict(self, model_name, steps:int=0, format_to_string:bool=True):
        if steps == 0: raise Exception('Steps must be greater than 0')
        forecast = self.models[model_name].forecast(steps=steps)

        dates = self.get_forecast_dates(steps)
        forecast = forecast.reset_index(drop=True)
        forecast = forecast.to_frame(name=model_name)
        forecast.insert(0, 'date', dates)
        if format_to_string:
            forecast[model_name] = forecast[model_name].apply(lambda x: '{:.6f}'.format(x))
        return forecast

if __name__ == "__main__":
    predict = Utils('costco').predict
    forecast = predict('sales_model', 10)
    print(forecast)
