import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
import time

def processImage(originalPath, skinPath, imageName, blockSideSize=7, resizeTo = 0.1):

    import numpy as np
    import math
    import PIL.Image 
    
    #imagePath  = "Users/mvijayen/bda/Original/train/"+imageName
    #skinPath = "Users/mvijayen/bda/Skin/train/"+imageName[0:-4]+"_s.bmp"
    imagePath  = originalPath+imageName
    skinPath = skinPath+imageName[0:-4]+"_s.bmp"
    skin = PIL.Image.open(skinPath)
    image = PIL.Image.open(imagePath)
    imSizeX,imSizeY = image.size

    if resizeTo !=  None:
        imSizeX = int(resizeTo*imSizeX)
        imSizeY = int(resizeTo*imSizeY)
        resizedSize= imSizeX, imSizeY
        image.thumbnail(resizedSize,PIL.Image.ANTIALIAS)
        skin.thumbnail(resizedSize, PIL.Image.ANTIALIAS)
        imSizeX,imSizeY = image.size
    
    pixels = image.load()
    skinPixels = skin.load()                          
    imSizeRGB = 3

    fringePixels = np.int(math.floor(blockSideSize/2)) # to be ignored, use sizeOfBlock x sizeOfBlock blocks
    columnSampleSize = imSizeY - fringePixels*2
    rowSampleSize = imSizeX - fringePixels*2
    numOfSamples=(imSizeX-fringePixels*2)*(columnSampleSize) #eg. if x side of image(and y is 7) is 10 then only have 4 samples 

    numOfColsPerPartialSample =  blockSideSize*blockSideSize*3
    partialSample = np.zeros((1,numOfColsPerPartialSample))
    partialSamples = np.zeros((numOfSamples,numOfColsPerPartialSample))

    numOfColumnsPerSample = numOfColsPerPartialSample+1 # the cube made by the block and rbg + 1 for class label of skin
    sample = np.zeros((1,numOfColumnsPerSample))
    samples = np.zeros((numOfSamples,numOfColumnsPerSample))

    isSkin = 1
    currentSample=0

    for x in  range(0+fringePixels,imSizeX-fringePixels):
        for y in range(0+fringePixels,imSizeY-fringePixels):
            if ((np.array(skinPixels[x,y])==255).all()):
                    isSkin=np.uint8(0)
            partialSampleIndex = 0
            for blockFringe in range(fringePixels, 0,-1): #use fringes in order have the spiral order, skip if fringe is 0                      
                xBlock = x-blockFringe   # top section of outer block
                for yBlock in range(y-blockFringe,y+blockFringe+1):#+1 because it excludes last index
                    partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])                        
                    partialSampleIndex =partialSampleIndex+imSizeRGB #next 3 RGB values
                yBlock = y+blockFringe #middle section right 
                for xBlock in range(x-(blockFringe-1),x+blockFringe):
                    partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])
                    partialSampleIndex = partialSampleIndex+imSizeRGB
                yBlock = y-blockFringe  #middle section left
                for xBlock in range(x-(blockFringe-1),x+blockFringe):
                    partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])
                    partialSampleIndex = partialSampleIndex+imSizeRGB
                xBlock = x+blockFringe #bottom section
                for yBlock in range(y-blockFringe,y+blockFringe+1):
                    partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[xBlock,yBlock])
                    partialSampleIndex = partialSampleIndex+imSizeRGB
            partialSample[0,partialSampleIndex:partialSampleIndex+imSizeRGB] = np.array(pixels[x,y])#for the middle point (same as when block side size is 1) 
            sample[0,0:numOfColsPerPartialSample]=partialSample
            sample[0,numOfColumnsPerSample-1] = isSkin
            samples[currentSample,:]= sample
            currentSample=currentSample+1;
            isSkin=1;
    
    return (np.asarray(samples, dtype=np.uint8))


def pixelArray (originalPath, skinPath, imgNames):
    for i,fname in enumerate(imgNames):
        if i==0:
            pixels = processImage(originalPath, skinPath, fname)
        else:
            pixels = np.concatenate((pixels,processImage(originalPath, skinPath, fname)), axis=0)
    return pixels
    

originalTrainPath = "/Shared/bdagroup5/Original/train/"
skinTrainPath = "/Shared/bdagroup5/Skin/train/"
imgTrainNames = [f for f in os.listdir(originalTrainPath) if not f.startswith('.')]
img_train = pixelArray(originalTrainPath, skinTrainPath, imgTrainNames)
train_shape = img_train.shape
X_train = img_train[:,0:-1]
Y_train = img_train[:,-1]
knn_fit_start = time.time()
knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', metric='minkowski', p=2, metric_params=None)    
knn.fit(X_train,Y_train)
knn_fit_end = time.time()
fit_time = knn_fit_end - knn_fit_start
print fit_time
print train_shape
print "X and Y train complete"

originalValPath = "/Shared/bdagroup5/Original/val/"
skinValPath = "/Shared/bdagroup5/Skin/val/"
imgValNames = [f for f in os.listdir(originalValPath) if not f.startswith('.')]
imgValNames = imgValNames[0:2]
print imgValNames

with open("/Shared/bdagroup5/kNNprocessed/k1bs7.txt",'a') as fn:
    for i in imgValNames:
        print i
        img_val = processImage(originalValPath, skinValPath, i)
        X_val = img_val[:,0:-1]
        Y_val = img_val[:,-1]
        knn_predict_start = time.time()
        predictYval = knn.predict(X_val)
        knn_predict_end = time.time()
        predict_time = knn_predict_end - knn_predict_start
        print predict_time
        knn_time = fit_time + predict_time
        percent_accuracy = sklearn.metrics.accuracy_score(Y_val, predictYval, normalize=True, sample_weight=None)*100
        cm = sklearn.metrics.confusion_matrix(Y_val, predictYval)
        cm_vals = np.concatenate(([cm[0][0]], [cm[0][1]], [cm[1][0]], [cm[1][1]], [percent_accuracy], [knn_time]))
        print cm_vals
        np.savetxt(fn ,cm_vals[None],fmt='%d, %d %d %d %f %f')

#tp = cm[0][0], fp = cm[0][1], fn = cm[1][0], tn = cm[1][1]
#f = open("/Users/mvijayen/bda/processed/k1bs7.txt",'a')
#img_val = os.listdir("/Users/tdogan/Desktop/Original/validate/")
