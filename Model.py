import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import argparse
from sklearn.svm import SVC
from sklearn.externals import joblib 


def model_train(df):
    print(df)
    feature_set  = df.iloc[:,0:5].to_numpy()
    class_labels = df['class'].to_numpy()
    print(feature_set)
    print(class_labels)
    model = SVC(gamma='auto')
    model.fit(feature_set, class_labels)
    joblib.dump(model, 'svm.pkl') 


def pcaing(df, fn, components = ''):
    print("PCAing is running for file".format(fn))
    print("df ",df.shape)
    scaled_df=StandardScaler().fit_transform(df)
    print("scaled_df",scaled_df.shape)
    f_mat = pd.DataFrame(data=scaled_df)

    if components == '':
        print("Components were not defined...\nCalculating...")
        pca = PCA(n_components=5)
    
        principalComponents = pca.fit_transform(scaled_df)
    
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4','principal component 5'])
        #principalDf.to_csv('/home/hp/Documents/CSE 572/features/PCA/pca_features_5.csv')
   
        vt = pca.components_

        components = vt.transpose()
        components_to_file = pd.DataFrame(data = components)
        components_to_file.to_csv('pca_components.csv', index = False, header = False)
       
    else:
        print("Wow... Components were defined")

    print(f_mat.shape, components.shape)
    f_mat_new = np.dot(f_mat, components)

    components_df= pd.DataFrame(data = f_mat_new)
    components_df.to_csv('pca_components_{}'.format(fn), index = False, header = False)
    print("output files: {}".format('pca_components_{}'.format(fn)))
    return components, components_df
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meal_folder"   , '-mf' , dest = "meal_folder", required = True, help ="A folder containing meal feature files")
    parser.add_argument("--no_meal_folder", '-nmf' , dest = "no_meal_folder", required = True, help = "A folder containing no_meal feature files")
    return parser.parse_args()

def main():
    options = parse_arguments()
    onlyfiles = [f for f in listdir(options.meal_folder) if isfile(join(options.meal_folder, f))]
    
    components = None
    cwd = os.getcwd()
    os.chdir(options.meal_folder)    

    for f in onlyfiles:
        print("file name is :" + f)
        df=pd.read_csv(f)
        #print(df)
        fn = f.replace("_features", "")
        components, df_meal = pcaing(df, fn)
    
    onlyfiles = [f for f in listdir(options.no_meal_folder) if isfile(join(options.no_meal_folder, f))]
    
    os.chdir(options.no_meal_folder)    
    for f in onlyfiles:
        print("file name is :" + f)
        df=pd.read_csv(f)
        #print(df)
        fn = f.replace("_features", "")
        components, df_no_meal = pcaing(df, fn, components)

    os.chdir(cwd)    

    #Training begins
    label_meal = np.array([1]*len(df_meal))
    label_nomeal = np.array([0]*len(df_no_meal))
    df_meal['class'] = label_meal
    df_no_meal['class'] = label_nomeal

    model_train(pd.concat([df_meal,df_no_meal]))

if __name__ == '__main__':
    main()
