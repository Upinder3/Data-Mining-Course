import csv
import pickle
import sys
import numpy as np
import pandas as pd
from numpy import trapz, ma
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import argparse

def extract_labels(inputFile):
    label = []
    classLabel = []
    with open(inputFile) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            label.append(row[3:10])
            classLabel.append(int(row[-1]))
    return label, classLabel


def featureExtraction(x_test):
    def moving_avg(data, size):
        weights = np.repeat(1.0, size) / size
        ma = np.convolve(data, weights, 'valid')
        return ma

    # creating features for time-series
    feature_set = []
    df = pd.DataFrame(x_test)
    for i in range(0,df.shape[0]):

        row = df.iloc[[i]]
        row_arr = row.to_numpy()
        result = []

        for j in range(len(row_arr[0]) - 1):
            calc_area = trapz([row_arr[0][j], row_arr[0][j + 1]], dx=5)
            result.append(calc_area)

        for j in range(len(row_arr[0]) - 1):
            calc_velocity = (row_arr[0][j + 1] - row_arr[0][j]) / 5
            result.append(calc_velocity)

        rfft = np.fft.rfft(row_arr[0])

        rfft_log = np.log(np.abs(rfft) ** 2 + 1)

        result.extend(moving_avg(row_arr[0], 2))

        result.extend(rfft_log)
        feature_set.append(result)
    df_feature = pd.DataFrame(feature_set)
    scaled_df = StandardScaler().fit_transform(df_feature)
    return scaled_df


def testModel(model_file, data):
    loaded_model = joblib.load(model_file)
    ypred = loaded_model.predict(data)
    return ypred


def csvToArray(pcaComponents):
    results = []
    with open(pcaComponents) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)
    return results

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file"    , '-i' , dest = "inputFile", required = True, help="A file containing test samples (with labels)")
    parser.add_argument("--model"         , '-m' , dest = "modelFile", default = 'svm.pkl', help="A file with loaded model")
    parser.add_argument("--pca_components", '-pc', dest = "pcaComponents_file", default = 'pca_components_meal.csv', help="A file containing test samples (with labels)")
    return parser.parse_args()

def main():
    options = parse_arguments()

    x_test, y_test = extract_labels(options.inputFile)
    df = featureExtraction(x_test)
    pca = csvToArray(options.pcaComponents_file)
    f_mat = np.dot(df, pca)
    ypred = testModel(options.modelFile, f_mat)
    print("Predicted Values: ",ypred)
    print(classification_report(y_test, ypred))

if __name__ == '__main__':
    main()
