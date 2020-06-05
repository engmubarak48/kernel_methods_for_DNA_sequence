"""
Created on Sun May 31 11:12:25 2020

@author: Jama Hussein Mohamud
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from kernels_models import *
from pandarallel import pandarallel
import argparse

pandarallel.initialize()
#%%


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m", "--model_name", required=False, help="name of the model", default='MKSM', type=str)
ap.add_argument("-c", "--C", required=False, help="C parameter for SVM", default=0.1, type=float)
ap.add_argument("-n_k", "--n_k", required=False, help="Kernels to combine; for spectrum kernel", default=[4,5,6], type=list)
ap.add_argument("-mk", "--matrix_kernel", required=False, help="kernel matrix to use; Gram_matrix_base or Gram_matrix_baseK")
ap.add_argument("-n", "--normalise", required=False, help="Normalise kernel matrix", default=True)
ap.add_argument("-k", "--kernel", required=False, help="kernel for KernelLogisticRegression", default='rbf')
ap.add_argument("-s", "--sigma", required=False, help="sigma for KernelLogisticRegression", default=2.28, type=float)
ap.add_argument("-mmk", "--mmk", required=False, help="K for the mismatch kernel", default=11, type=int)
ap.add_argument("-mm", "--mm", required=False, help="number of mismatches", default=1, type=int)
ap.add_argument("-sp", "--split_size", required=False, help="split size", default=0.8, type=float)

args = vars(ap.parse_args())

#%%


try:
    train_data = pd.read_csv('Xtr.csv')
    test_data = pd.read_csv('Xte.csv')
    y_train = pd.read_csv('Ytr.csv')
    y = np.array(y_train['Bound'])
    y[y == 0] = -1
    y[y == 1] = 1
    
    print('train shape: ', train_data.shape, 'test shape: ',test_data.shape, 'Labels shape: ', y_train.shape)
except:
    print('please make sure the script and data files are in the same directory')
# Uncomment to plot the bar plot
#y_train['Bound'].value_counts().sort_index().plot.bar()

#%%



def split_data(train_data, y, split_size = 0.8):
    print('------ spliting data -----------')
    tr_size = int(len(train_data) * split_size)
    X_train, X_test, y_train, y_test = train_data.iloc[:tr_size], train_data.iloc[tr_size:], y[:tr_size], y[tr_size:]
    return X_train, X_test, y_train, y_test

def transform(data, transform_type='PCA'):
    if transform_type=='PCA':
        transformed = PCA(n_components=2).fit_transform(np.array(data))
    if transform_type=='SVD':
        svd = TruncatedSVD(n_components=400, n_iter=20, random_state=24)
        transformed = svd(n_components=2).fit_transform(np.array(data))
    return transformed

def display(reduced, labels):
  plt.figure(dpi=100)
  for l in range(2):
    these_points = reduced[labels == l]
    plt.scatter(these_points[:, 0], these_points[:, 1], label=str(l))
  plt.legend(bbox_to_anchor=(1, 1), loc=2)
  plt.xlabel('D1')
  plt.ylabel('D2')
  plt.show()

def concat_data(data1, data2):
    concated_data = pd.concat([data1, data2], axis=1)
    return concated_data

# Prediction Accuracy 
def Accuracy(ytrue, ypred):
    return np.mean(ypred == ytrue)

def train_fun(data, label, model):
    model.fit(data, label)
    return model

def prediction_fun(model, test_data, filename):
    print('--------- predicting on test data -------- ')
    predictions_sub = model.predict(test_data)
    predictions_sub[predictions_sub == -1] = 0
    predictions_sub[predictions_sub == 1] = 1
    predictions_sub = np.array(predictions_sub, dtype=int)
    
    print('-------- Generating submission file ----------')
    sub_ran = {'Id':test_data.Id.values,'Bound': predictions_sub}
    submission = pd.DataFrame.from_dict(sub_ran, orient='columns')
    
    file_name = filename
    
    submission.to_csv(f'{file_name}', index=False)
    
    print('END: submission file generated as:', file_name)
    return submission


# display transformed data in 2-D
#display(transformed, np.array(y_train['Bound']))

model_name = args["model_name"] 
C = args["C"] 
k = args["n_k"] 
matrix_kernel = args['matrix_kernel']
normalise = args["normalise"]  
kernel = args["kernel"]
sigma = args["sigma"]
mmk = args["mmk"]
mm = args["mm"]
split_size = args["split_size"]

coeff = (1/(len(k)+1))*np.ones(len(k)+1)

#models = {'MKSS': Multi_SpectrumSVM(C=1, k=k, coeff = coeff, matrix_kernel=Gram_matrix_base, normalise=True),
#          'KLR': KernelLogisticRegression(lambd=1, kernel=kernel, sigma=sigma),
#          'MKSM': KernelSVM(C=1, k=mmk, m=mm)}

if model_name == 'MKSS':
    model = Multi_SpectrumSVM(C=C, k=k, coeff = coeff, matrix_kernel=Gram_matrix_base, normalise=True)
elif model_name == 'KLR':
    model = KernelLogisticRegression(lambd=1, kernel=kernel, sigma=sigma)
elif model_name == 'MKSM':
    model = KernelSVM(C=1, k=mmk, m=mm)
    

print('------ training on spectrum kernels ------------')

X_train, X_test, y_train, y_test = split_data(train_data, y, split_size)

model = train_fun(X_train, y_train, model)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = Accuracy(y_train, train_pred)
test_accuracy = Accuracy(y_test, test_pred)
print(f'Train accuracy: {train_accuracy:.2f}')
print(f'Valid accuracy: {test_accuracy:.2f}')


file_name = 'submission.csv'

prediction_fun(model, test_data, file_name)





