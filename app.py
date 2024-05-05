from flask import Flask, render_template, request
import pickle
import numpy as np
import keras_model
# import default libraries
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
from sklearn import metrics
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import keras_model
########################################################################
import librosa

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
# model = pickle.load(open('iri.pkl', 'rb'))


app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    # data1 = request.form['a']
    # data2 = request.form['b']
    section_idx = request.form['a']
    machine_type = request.form['b']
    file = request.files['file']
    # print(section_idx)
    # arr = np.array([[data1, data2, data3, data4]])
    # load anomaly score distribution for determining threshold
    if machine_type == "fan":
       model = keras_model.load_model("model_fan.hdf5")
    elif machine_type == "pump":
       model = keras_model.load_model("model_pump.hdf5")
    if machine_type == "fan":
       score_distr_file_path = "score_distr_fan.pkl"
       shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)
       # determine threshold for decision
       decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)
    elif machine_type == "pump":
        score_distr_file_path = "score_distr_pump.pkl"
        shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)
        # determine threshold for decision
        decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

# Assuming 'file_data' contains the wave file data
    y, sr = librosa.core.load(file, sr=None, mono=True)
    data = com.file_to_vectors(y,sr,
                                                        n_mels=param["feature"]["n_mels"],
                                                        n_frames=param["feature"]["n_frames"],
                                                        n_fft=param["feature"]["n_fft"],
                                                        hop_length=param["feature"]["hop_length"],
                                                        power=param["feature"]["power"])
    condition = np.zeros((data.shape[0], 2), float)
    # pred = model.predict(arr)
    # condition = np.zeros((data.shape[0], n_sections), float)
                    # if the id_name was found in the trained_section_names, make a one-hot vector
    # if section_idx != -1:
    #     condition[:, section_idx : section_idx + 1] = 1

    # 1D vector to 2D image
    data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)
    
    p = model.predict(data)[:, int(section_idx): int(section_idx) + 1]
    y_pred = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon) 
                               - np.log(np.maximum(p, sys.float_info.epsilon))))
    decision_result = 0
    if y_pred > decision_threshold:
        decision_result =1
    else:
        decision_result =0
    print(decision_threshold)
    return render_template('after.html', data=decision_result)


if __name__ == "__main__":
    app.run(debug=True)















