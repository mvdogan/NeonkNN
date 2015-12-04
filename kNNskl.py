import numpy as np
import sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import time

f = open("/Users/tdogan/Desktop/k1bs7.txt",'a')
img_train = np.loadtxt("/Users/tdogan/Desktop/test.txt")
img_val = np.loadtxt("/Users/tdogan/Desktop/test2.txt")
X_train = img_train[:,0:-1]
Y_train = img_train[:-1]
X_val = img_val[:,0:-1]
Y_val = img_val[:-1]
knn_start = time.time()
knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', metric='minkowski', p=2, metric_params=None)    
knn.fit(X_train,Y_train)
predictYval = knn.predict(X_val)
knn_end = time.time()
knn_time = knn_end - knn_start
percent_accuracy = sklearn.metrics.accuracy_score(Y_val, predictYval, normalize=True, sample_weight=None)*100
cm = sklearn.metrics.confusion_matrix(Y_val, predictYval)
#tp = cm[0][0]
#fp = cm[0][1]
#fn = cm[1][0]
#tn = cm[1][1]
cm_vals = np.concatenate(([cm[0][0]], [cm[0][1]], [cm[1][0]], [cm[1][1]], [knn_time]))
np.savetxt(f,cm_vals[None],fmt='%d')
f.close()
