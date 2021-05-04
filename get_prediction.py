import json, requests, numpy
from app import Model_train
from keras.datasets import cifar10

def get_prediction(X):
    data = {'data': X.tolist() if isinstance(X, numpy.ndarray) else X}
    # res = requests.post('http://0.0.0.0:8002/getPrediction', data=json.dumps({'data': data}))
    res = requests.post('http://127.0.0.1:8002/getPrediction', data=json.dumps(data))
    return res.json()


if __name__ == '__main__':
    model = Model_train()
    print('true:', numpy.argmax(model.y_test[0]),
          '\npred:', get_prediction(model.x_test[0])['prediction'])