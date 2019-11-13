from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from scipy.fftpack import fft, ifft
import pandas as pd
import math
from os import listdir
from os.path import isfile, join
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", '-f' , dest = "folder", required = True, help="A folder containing files to clean")
    return parser.parse_args()

def main():
    options = parse_arguments()
    folder = options.folder
    os.chdir(folder)
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    
    for f in onlyfiles:
        if 'csv' not in f:
            continue 
        print("Cleaning file: ",f)
    
        df=pd.read_csv(f)
    
        final_clean=[]
        for i in range(0,df.shape[0]):
            row=df.iloc[[i]]
            row_arr=row.to_numpy()
            record=pd.Series(row_arr[0])
            processed=record.interpolate(method='spline', order=2,limit_direction='both')
            final_clean.append(np.floor(np.abs((processed.to_numpy()))))
        
        final_clean=np.asarray(final_clean)
        df_cleaned=pd.DataFrame(final_clean)
    
        print("File is cleaned!")

if __name__ == '__main__':
    main()
