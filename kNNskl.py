def NeonKNN(k=1, block_size=7, logfile='logfile.txt'):

    import os
    import csv
    import time
    import numpy as np
    import sklearn.metrics
    from scipy import stats
    from sklearn.neighbors import KNeighborsClassifier
    from utils import processImage, pixelArray, _match_class

    # Path
    originalTrainPath = "/Shared/bdagroup5/Original/train/"
    skinTrainPath = "/Shared/bdagroup5/Skin/train/"

    # Load training samples
    imgTrainNames = [f for f in os.listdir(originalTrainPath) if not f.startswith('.')]
    #imgTrainNames = imgTrainNames[0:199]
    train_pi_start = time.time()
    img_train = pixelArray(originalTrainPath, skinTrainPath, imgTrainNames, blockSize=block_size)
    train_pi_end = time.time()
    train_pi_time = train_pi_end - train_pi_start
    #train_shape = img_train.shape
    X_train = img_train[:,0:-1]
    Y_train = img_train[:,-1]
    del X_train

    # Timing: start
    knn_fit_start = time.time()

    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', metric='minkowski', p=2, metric_params=None)    
    fitting = knn.fit(X_train,Y_train)
    knn_fit_end = time.time()
    fit_time = knn_fit_end - knn_fit_start
    #print fit_time
    #print train_shape
    print "X and Y train complete"

    # Load validation samples
    originalValPath = "/Shared/bdagroup5/Original/val/"
    skinValPath = "/Shared/bdagroup5/Skin/val/"
    imgValNames = [f for f in os.listdir(originalValPath) if not f.startswith('.')]
    # control the number
    #imgValNames = imgValNames[0:2]
    #print imgValNames

    counter = 1
    with open("/Shared/bdagroup5/kNNprocessed/{}".format(logfile),'w') as fn:
        # writer for this filename fn
        w = csv.writer(fn)
        for i in imgValNames:
            print 'Working on the {}th images!'.format(counter)
            #print i,
            img_val = processImage(originalValPath, skinValPath, i, blockSideSize=block_size)
            X_val = img_val[:,0:-1]
            Y_val = img_val[:,-1]
            knn_predict_start = time.time()
            dist = fitting.kneighbors(X_val,k,return_distance=True)
            del X_val
            #predictYval = knn.predict(X_val)
            #upscaleBinary(predictYval,i,0.1)
            knn_predict_end = time.time()
            predict_time = knn_predict_end - knn_predict_start
            #print predict_time,
            knn_time = fit_time + predict_time
            #percent_accuracy = sklearn.metrics.accuracy_score(Y_val, predictYval, normalize=True, sample_weight=None)*100
            #cm = sklearn.metrics.confusion_matrix(Y_val, predictYval)        
            #cm_vals = np.concatenate(([cm[0][0]], [cm[0][1]], [cm[1][0]], [cm[1][1]], [percent_accuracy], [knn_time]))

            # Use csv.writerow()
            #w.writerow([i] + cm.flatten().tolist() + [percent_accuracy, knn_time])

            # Match the indices to their corresponding classes
            dist_array, ind_array = dist[0], dist[1]
            cls_array = _match_class(Y_train, ind_array)

            output_row = [i, knn_time] 

            # for each k, get confusion matrix
            for ki in range(1, k+1, 2):
                sub_cls_array = cls_array[:, :ki]
                # find the mode in each row
                _pred = stats.mode(sub_cls_array, axis=1)[0].flatten().tolist()
                cm = sklearn.metrics.confusion_matrix(Y_val, _pred)        
                # output the flattened confusion matrix
                output_row += cm.flatten().tolist()

            # headers: Image name; Confusion entries for all k's; train tim
            w.writerow(output_row)

            counter += 1

            #np.savetxt(fn ,cm_vals[None],fmt='%d, %d %d %d %f %f')
            #print cm_vals


