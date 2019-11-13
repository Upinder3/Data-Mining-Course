#!/usr/bin/python3
import numpy as np
from numpy import ma, trapz
import pandas as pd

import readline
import csv
import openpyxl

from scipy.fftpack import fft, ifft
import os
from os import listdir
from os.path import isfile, join
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mealfolder", '-mf' , dest = "meal_folder", required = True, help = "A meal folder containing files meal files")
    parser.add_argument("--nomealfolder", '-nmf' , dest = "nomeal_folder", required = True, help = "A no meal folder containing files no meal files")
    return parser.parse_args()


def moving_avg (data, size):
    weights = np.repeat(1.0, size)/size
    ma = np.convolve(data, weights, 'valid')
    return ma


def extract_features(folder):
    os.chdir(folder)
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    
    print("Extracting features from file below file:\n{}".format(onlyfiles))

    #creating features for time-series
    
    for f in onlyfiles:
        feature_set=[]
        #Read the file
        df = pd.read_csv(f)
        for i in range(0,df.shape[0]):
            row = df.iloc[[i]]
            row_arr = row.to_numpy()
            result = []
    
            for j in range(len(row_arr[0]) - 1):
                calc_area = trapz([row_arr[0][j], row_arr[0][j+1]], dx=5)
                result.append(calc_area)
        
            for j in range(len(row_arr[0]) - 1):
                calc_velocity = (row_arr[0][j+1] - row_arr[0][j])/5
                result.append(calc_velocity)
        
            rfft = np.fft.rfft(row_arr[0])
            
            rfft_log = np.log(np.abs(rfft) ** 2 + 1)
        
            result.extend(moving_avg(row_arr[0],2))
        
            result.extend(rfft_log)
            feature_set.append(result)
    
        df_feature=pd.DataFrame(feature_set)
        print(df_feature.shape)
        df_feature.to_csv('features_{}'.format(f), header=False, index=False)

def main():
    options = parse_arguments()
    
    extract_features(options.meal_folder)
    extract_features(options.nomeal_folder)

if __name__ == '__main__':
    main()
