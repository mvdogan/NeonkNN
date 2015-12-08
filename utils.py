def _match_class(y, index_matrix):
    '''
        Internal method to match the index for the class
    '''
    import numpy.matlib
    output = numpy.matlib.zeros(index_matrix.shape, dtype=np.uint8)
    index_matrix = np.asmatrix(index_matrix)
    for row in index_matrix.shape[0]:
        # row
        indices = index_matrix[row, :]
        output[row] = y[indices]

    return np.asarray(output)

def processImage(originalPath, skinPath, imageName, blockSideSize=7, resizeTo = 0.1):
    '''
        Preprocess Image
    '''

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
            currentSample=currentSample+1
            isSkin=1
    
    return (np.asarray(samples, dtype=np.uint8))

def processImageRowTest(imageName, blockSideSize=7, resizeTo = None, test = 0):
    '''
        Update on 12/05/15: Change the resize feature to be a ratio
    '''

    import numpy as np
    import math
    import PIL.Image 
    
    imagePath  = "../Original/train/"+imageName
    skinPath = "../Skin/train/"+imageName[0:-4]+"_s.bmp"
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
            if test==1:
                partialSamples[currentSample,:] = partialSample
            else:
                sample[0,numOfColumnsPerSample-1] = isSkin
                samples[currentSample,:]= sample
                isSkin=1;
            currentSample=currentSample+1;
            
    if test==1:
        samples = partialSamples
    
    return (np.asarray(samples, dtype=np.uint8))

def pixelArray (originalPath, skinPath, imgNames, blockSize=7):
    import numpy as np
    for i,fname in enumerate(imgNames):
        if i==0:
            pixels = processImage(originalPath, skinPath, fname, blockSideSize=blockSize)
        else:
            pixels = np.concatenate((pixels,processImage(originalPath, skinPath, fname, blockSideSize=blockSize)), axis=0)
    return pixels

def originalResize(createResizedImages = 0):
    import PIL.Image
    import os
    import numpy as np

    #different images paths with images to shrink. Assumes that it exists
    imagePaths = list(["/Original/val/","/Original/train/","/Original/test/","/Skin/val/","/Skin/train/"])

    #resizings for images
    resizeRange = np.arange(0.1,1.1,0.1)

    #create folder structure for where the resized images will reside
    for imagePath in imagePaths:
        for resize in resizeRange:
            resizedPath = "/Shared/bdagroup5"+str(int(resize*10))+imagePath
            if not os.path.exists(resizedPath):
                os.makedirs(resizedPath)

    # to output the original sizes since resize does truncating
    # and thumbnail can give other values than the ones
    # provided(fixes one) when resizing
    textFile = open("/Shared/bdagroup5/OriginalImageSizes.txt","w")

    originalCount = 0

    #deposit shrunk images in this directory
    for imagePath in imagePaths:
        imageSubset = os.listdir("/Shared/bdagroup5"+imagePath)
        for imageName in imageSubset:   
            imageOriginal = PIL.Image.open("/Shared/bdagroup5"+imagePath+imageName)
            originalWidth, originalHeight = imageOriginal.size
            textFile.write("{0}\t{1}\t{2}\n".format(imageName,originalWidth,originalHeight))
            if createResizedImages == 1:
                for resize in resizeRange:
                    imageCopy = imageOriginal.copy()
                    if resize==1.0:
                        imageCopy = imageOriginal
                        width = originalWidth
                        height = originalHeight
                    else:
                        imageCopy.thumbnail((originalWidth*resize,originalHeight*resize),PIL.Image.ANTIALIAS)
                        width, height = imageCopy.size
                    resizedImagePath = "/Shared/bdagroup5/"+str(int(resize*10))+imagePath+imageName
                    if originalCount < 3:
                        imageCopy.save(resizedImagePath,"JPEG") 
                    else:
                        imageCopy.save(resizedImagePath,"BMP")
        originalCount = originalCount + 1            
    textFile.close()


def upscaleBinary(oneDArray,imageName,resize):
    import PIL.Image
    import math

    originalResize()
    origImageSizes = open("/Shared/bdagroup5/OriginalImageSizes.txt","r")
    data = origImageSizes.readlines()
    collectImageDimensions = []
    for eachImage in data:
        collectImageDimensions.append(eachImage[:-1].split('\t'))

    original = collectImageDimensions[np.where(np.array(collectImageDimensions) == imageName)[0][0]]
    binaryPredicted = PIL.Image.new('L', (math.floor(int(original[1])*resize),math.floor(int(original[2])*resize))) 
    binaryPredicted.putdata(oneDArray)
    originalImage = binaryPredicted.resize(int(original[1]),int(original[2]),PIL.Image.ANTIALIAS)
    bw.save("/Shared/bdagroup5/greyscaleImages/predicted_"+imageName)
