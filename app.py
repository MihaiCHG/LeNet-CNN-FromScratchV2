from utils.LayerObjects import *
from utils.utils_func import *
from flask import Flask, jsonify, render_template, request
from preprocessing import *
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/digit_prediction', methods=['POST'])

def digit_prediction():
    if(request.method == "POST"):
        img = request.get_json()
        img = preprocess(img)
        test_image_normalized_pad = normalize(zero_pad(img[:,:,:,np.newaxis], 2), 'lenet5')
        probability, digit = ConvNet.Forward_Propagation(test_image_normalized_pad, 1, 'test_single')
        print(digit)
        print("error rate:", probability / len(digit))
        data = { "digit":int(digit), "probability":float(int(probability*100))/100. }
        return jsonify(data)

if __name__ == "__main__":
    ConvNet = LeNet5()
    with open('model_data_19.pkl', 'rb') as input_:
        ConvNet = pickle.load(input_)
    app.run(debug=True)
