import pandas as pd
import cv2
import seaborn as sb

import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import zipfile
from scipy import ndimage

label_csv = pd.read_csv('public_list_primary_v3_testset_final.csv') #labels
label_csv.index = label_csv.UUID #make ID the index
location = 'data_out/sanjoy_seismogram'

def seismogram(dat_file):
    '''
    Sanjoy's Seismogram image processing function
    Create binary image of highly denoised spectrogram
    '''
    return ndimage.imread(os.path.join(location, dat_file))

#linear (r^2) classifier for narrowband/DRD
def narrow_linear_fit(image):

    '''
    arugments:  the image file name

    this function applies image processing for smoothing/closing
    it then skeletonizes the image to a line width of one
    these pixel locations are taken and fit to a linear model
    r^2 is taken to distinguish between narrowband (linear) and narrowbandrd (curved)

    returns: linear model object, dependent values for line, independent values for line, spectrogram
    '''

    # IMAGE PROCESSING #
    spectrogram = seismogram(image)
    spectrogram = np.uint8(spectrogram) #prep for openCV
    kernel = np.ones((11,11),np.uint8) #kernel size for morpho closing
    spectrogram = cv2.morphologyEx(spectrogram, cv2.MORPH_CLOSE, kernel) #close holes
    spectrogram = cv2.GaussianBlur(spectrogram, (11,11), 0) #close gaps
    spectrogram[spectrogram > 0] = 255 #make binary

    #skeletonize the band, put line loc into curve list
    def skeletonize(row):
        return np.median(np.array(np.nonzero(row > 0)))
    curve = []
    curve.append(np.apply_along_axis(skeletonize, 1, spectrogram))
    curve =  np.array(curve).flatten() #convert to numpy for speed
    curve = curve[~np.isnan(curve)] # get rid of NAN values (where image is blank)
    curve_std = np.std(curve) #standard deviation of curve values

    #scale data between 0-~32 (height of raw image, minus the dropped NANs)
    scaler = MinMaxScaler()
    curve = scaler.fit_transform(curve.reshape(-1, 1))
    X = np.linspace((curve.shape[0]-1), 0, curve.shape[0]).flatten()
    curve = curve*X.shape[0]

    #constrain  edges (seemed to cause some issues)
    curve = curve[2:-2]
    X = X[2:-2]

    #fit a linear regression to the skeletonized line
    X = sm.add_constant(X)
    linear_fit = sm.OLS(curve, X).fit()
    slope, intercept = linear_fit.params

    return linear_fit, curve, X, slope, intercept, curve_std, spectrogram

'''#split data into features (images) and classes (labels)
full_df = pd.read_csv('flattend_square.csv')
labels_true = full_df.iloc[:, 0]
names = full_df.iloc[:, 1]
images = full_df.iloc[:, 2:]
images[images>0]=1  #convert back to binary (compression changed that)'''

#machine learning!
dirname = "data_out/sanjoy_seismogram/"
images = [scipy.misc.imresize(scipy.ndimage.imread(dirname + filename, flatten=True), 0.05) for filename in os.listdir(dirname)[:((int)(len(os.listdir(dirname))/4))]]
#get subsets of data to test around with
index_file = pd.read_csv(os.path.join('primary_medium', 'public_list_primary_v3_medium_21june_2017.csv'))
classifications = {}
for index, data in index_file.iterrows():
  classifications[data['UUID'] + '.dat.png'] = data['SIGNAL_CLASSIFICATION']
labels_true = [classifications[filename] for filename in os.listdir(dirname)]

dirname_test = "data_out/sanjoy_seismogram_test/"
images_test = [scipy.misc.imresize(scipy.ndimage.imread(dirname_test + filename, flatten=True), 0.1) for filename in os.listdir(dirname_test)]
#get subsets of data to test around with
index_file_test = pd.read_csv(os.path.join('primary_medium', 'public_list_primary_v3_medium_21june_2017.csv'))
classifications = {}
for index, data in index_file.iterrows():
  classifications[data['UUID'] + '.dat.png'] = data['SIGNAL_CLASSIFICATION']

#labels_test = labels_true_test #make copy of originals
labels = labels_true
#change labels for linreg classifier. We want narrowband and DRD to appear
#as the same class, so that the 1st pass classifier has an easier time distinguishing
def change_label(x):
    if x=='narrowbanddrd':
        return 'narrowband'
    else:
        return x

labels = [change_label(l) for l in labels]
div = len(images)
#train test split
trainX = images[:div]
testX = images_test
trainY = labels[:div]
#testY = labels_test
#trainY_true = labels_true
#testY_true = labels_true_test
trainX = np.stack(trainX, axis=0)
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1]*trainX.shape[2])
testX = np.stack(testX, axis=0)
testX = testX.reshape(testX.shape[0], testX.shape[1]*testX.shape[2])
#apply SVM (pretuned via grid search)
clf = svm.SVC(C = 10, gamma=0.01)
clf.fit(trainX, trainY)


#apply predictions on test data
#concat w/ true labels (untouched narrowbandDRD)
SVM_pred = clf.predict(testX)
names = [data['UUID'] for index, data in index_file_test.iterrows()]
#all_y = pd.union(frames, axis=1)
#all_y.columns =['testY_true', 'testY_SVM', 'prediction']
all_y = pd.DataFrame({'uuid': names, 'prediction': SVM_pred})
#whats our accuracy of the first pass classifier?
#when the narrowband and nrrowbandDRD were separate, this was 68%
#this is just to validate the accuracy of the 3 classes, where N and NDRD are combined
'''total = all_y.shape[0]
correct = 0
for i in all_y.index:
    if all_y.ix[i]['testY_SVM'] == all_y.ix[i]['prediction']:
        correct+= 1
print(correct/float(total))'''

#TO DO - CHANGE THIS TO A APPLY FUNCITION FOR SPEED

#apply linear regression/r^2 classifier
#change the predictions

r2_threshold = 0.996
for i in all_y.index:
    if all_y.ix[i]['prediction'] == 'narrowband': #if SVM returns narrowband, use linear classifier
        ID = all_y.ix[i]['uuid']+'.dat.png'
        model, line, x, m, b, std, spectrogram  = narrow_linear_fit(ID) #apply linear function
#         print model.rsquared_adj, std
        if  model.rsquared_adj > r2_threshold:
            prediction = 'narrowband'
        else:
            if std<5:     #add standard deviation conditional
                prediction = 'narrowband'
            else:
                prediction = 'narrowbanddrd'
        all_y.ix[i]['prediction'] = prediction

#get the new accuracy score
#were looking for an improvement on 68%

'''total = all_y.shape[0]
correct = 0
for i in all_y.index:
    if all_y.ix[i]['testY_true'] == all_y.ix[i]['prediction']:
        correct+= 1
print(correct/float(total))'''

#lets look at a confusion matrix to see where our errors are
labels = ["NAME_COL", "brightpixel", "narrowband", "narrowbanddrd", "noise", "squarepulsednarrowband", "squiggle", "squigglesquarepulsednarrowband"]
#cm = confusion_matrix(all_y['testY_true'], all_y['prediction'], labels=labels)
#cm = pd.DataFrame(data=cm, columns=labels, index=labels)
rows = []
for i in all_y.index:
  vect = [all_y.ix[i]['uuid'], 0.01,0.01,0.01,0.01,0.01,0.01,0.01]
  vect[labels.index(all_y.ix[i]['prediction'])] = 0.94
  rows.append(vect)

df2 = pd.DataFrame(rows)
df2.to_csv("mohit_output.csv", index=False, header=False)
