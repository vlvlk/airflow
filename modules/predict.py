import dill
import os
import pandas as pd
import json
import logging

path = os.environ.get('PROJECT_PATH', '..')

model_list = os.listdir(path + '/data/models/')
if model_list:
    model_path = path + '/data/models/' + sorted(model_list)[-1]
    with open(model_path, 'rb') as model_file:
        model = dill.load(model_file)
    logging.info('Model', sorted(model_list)[-1], 'loaded')


def predict_json(path_to_json):
    with open(path_to_json, 'rb') as data:
        data = pd.DataFrame(json.load(data), index=[0])
    data['price_category'] = model.predict(data)
    return data[['id', 'price_category']]


def predict():
    df = pd.DataFrame()
    files = os.listdir(path + '/data/test')
    try:
        for file in files:
            data = predict_json(path + '/data/test/' + file)
            df = pd.concat((df, data), axis=0)
        df.to_csv(path + '/data/predictions/result.csv', index=False)
        logging.info('Predictions saved as', path + '/data/predictions/result.csv')
    except:
        logging.warning('Model not loaded')


if __name__ == '__main__':
    predict()
