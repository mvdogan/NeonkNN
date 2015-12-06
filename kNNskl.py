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
    

originalTrainPath = "/Users/mvijayen/bda_project/Original/train/"
skinTrainPath = "/Users/mvijayen/bda_project/Skin/train/"
imgTrainNames = [f for f in os.listdir(originalTrainPath) if not f.startswith('.')]
img_train = pixelArray(originalTrainPath, skinTrainPath, imgTrainNames) 


originalValPath = "/Users/mvijayen/bda_project/Original/val/"
skinValPath = "/Users/mvijayen/bda_project/Skin/val/"
imgValNames = [f for f in os.listdir(originalValPath) if not f.startswith('.')]
img_val = pixelArray(originalValPath, skinValPath, imgValNames)


X_train = img_train[:,0:-1]
Y_train = img_train[:,-1]
X_val = img_val[:,0:-1]
Y_val = img_val[:,-1]
knn_start = time.time()
knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', metric='minkowski', p=2, metric_params=None)    
knn.fit(X_train,Y_train)
predictYval = knn.predict(X_val)
knn_end = time.time()
knn_time = knn_end - knn_start
percent_accuracy = sklearn.metrics.accuracy_score(Y_val, predictYval, normalize=True, sample_weight=None)*100
cm = sklearn.metrics.confusion_matrix(Y_val, predictYval)
cm_vals = np.concatenate(([cm[0][0]], [cm[0][1]], [cm[1][0]], [cm[1][1]], [percent_accuracy], [knn_time]))
np.savetxt("/Users/mvijayen/bda_project/processed/k1bs7.txt",cm_vals[None],fmt='%d')

#tp = cm[0][0], fp = cm[0][1], fn = cm[1][0], tn = cm[1][1]
#f = open("/Users/mvijayen/bda/processed/k1bs7.txt",'a')
#processImage(originalPath+original_train[0], skinPath+skin
#img_val = os.listdir("/Users/tdogan/Desktop/Original/validate/")
#img_train = np.loadtxt("/Users/tdogan/Desktop/test.txt")
#img_val = np.loadtxt("/Users/tdogan/Desktop/test2.txt")
