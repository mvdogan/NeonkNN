import os
import csv
import time
import numpy as np
import sklearn.metrics
from utils import processImage, pixelArray
from sklearn.neighbors import KNeighborsClassifier

# Path
originalTrainPath = "/Shared/bdagroup5/Original/train/"
skinTrainPath = "/Shared/bdagroup5/Skin/train/"

# Load training samples
imgTrainNames = [f for f in os.listdir(originalTrainPath) if not f.startswith('.')]
img_train = pixelArray(originalTrainPath, skinTrainPath, imgTrainNames)
train_shape = img_train.shape
X_train = img_train[:,0:-1]
Y_train = img_train[:,-1]

# Timing: start
knn_fit_start = time.time()

knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', metric='minkowski', p=2, metric_params=None)    
knn.fit(X_train,Y_train)
knn_fit_end = time.time()
fit_time = knn_fit_end - knn_fit_start
print fit_time
print train_shape
print "X and Y train complete"

# Load validation samples
originalValPath = "/Shared/bdagroup5/Original/val/"
skinValPath = "/Shared/bdagroup5/Skin/val/"
imgValNames = [f for f in os.listdir(originalValPath) if not f.startswith('.')]
# control the number
imgValNames = imgValNames[0:2]
print imgValNames

with open("/Shared/bdagroup5/kNNprocessed/k1bs7.txt",'a') as fn:
    # writer for this filename fn
    w = csv.writer(fn)
    for i in imgValNames:
        print i,
        img_val = processImage(originalValPath, skinValPath, i)
        X_val = img_val[:,0:-1]
        Y_val = img_val[:,-1]
        knn_predict_start = time.time()
        predictYval = knn.predict(X_val)
        upscaleBinary(predictYval,i,0.1)
        knn_predict_end = time.time()
        predict_time = knn_predict_end - knn_predict_start
        print predict_time,
        knn_time = fit_time + predict_time
        percent_accuracy = sklearn.metrics.accuracy_score(Y_val, predictYval, normalize=True, sample_weight=None)*100
        cm = sklearn.metrics.confusion_matrix(Y_val, predictYval)        
        cm_vals = np.concatenate(([cm[0][0]], [cm[0][1]], [cm[1][0]], [cm[1][1]], [percent_accuracy], [knn_time]))

        # Use csv.writerow()
        w.writerow([i] + cm.flatten().tolist() + [percent_accuracy, knn_time])
        #np.savetxt(fn ,cm_vals[None],fmt='%d, %d %d %d %f %f')
        print cm_vals

#tp = cm[0][0], fp = cm[0][1], fn = cm[1][0], tn = cm[1][1]
#f = open("/Users/mvijayen/bda/processed/k1bs7.txt",'a')
#img_val = os.listdir("/Users/tdogan/Desktop/Original/validate/")
