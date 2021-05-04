"""
Сервис предназначен для предcказания класса по датасету cifar10


author: vndanilchenko@gmail.com
"""


import flask, json, numpy
from app import Model_train


app = flask.Flask(__name__)


@app.route('/')
def home():
    return "Hello stranger! This Service predicts class for cifar10 dataset. Try POST request to /getPrediction " \
           "in format: {'data': (32, 32, 3) numpy.array}"


@app.route('/getPrediction', methods=['POST'])
def prediction():
    """
    эндпоинт для получения предсказания модели
    params: (32, 32, 3) numpy.ndarray
    :return: {'prediction': [class]} dict
    """
    params = flask.request.get_json(force=True, silent=True)

    if params and 'data' in params:
        res = model.predict(numpy.asarray(params['data']))
    else:
        res = None

    return flask.Response(response=json.dumps({'prediction': res}), status=200, content_type='application/json; charset=utf-8')


if __name__ == '__main__':
    model = Model_train()
    model.compile()
    app.run(host='127.0.0.1', port=8002, debug=True)