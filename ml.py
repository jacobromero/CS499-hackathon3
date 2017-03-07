from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from flask import Flask
from flask import request
import flask
from flask.ext.cors import CORS, cross_origin

data = [[10, 0, 1, 8], [10, 0, 2, 9], [10, 1, 1, 13], [5, 0, 1, 15], [5, 1, 5, 16], [5, 1, 6, 18]]
output = [[8], [7], [5], [6], [9], [9]]
pl = make_pipeline(PolynomialFeatures(1), linear_model.LogisticRegression())

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def InputPage():
    return flask.send_file("index.html")

@app.route("/input", methods=["POST"])
@cross_origin()
def train():
    p1 = request.args.get('p1')
    data.append([request.args.get('p1'), request.args.get('p2'), request.args.get('p3'), request.args.get('p4')])
    output.append([request.args.get('o1')])
    if (len(data) >= 2 and len(output) >= 2):
        pl.fit(data, output)
    return "ok"

@app.route("/predict", methods=["POST"])
@cross_origin()
def predictOutput():
    pred = [request.args.get('p1'), request.args.get('p2'), request.args.get('p3'), request.args.get('p4')]
    prediction = pl.predict([pred])
    return str(prediction[0])


if __name__ == "__main__":
    pl.fit(data, output)
    app.run()