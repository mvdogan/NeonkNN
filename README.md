# NeonkNN

Use `python main.py k block_size` to run scikit-learn kNN on Neon

Example: `python main.py 1 7` will be running kNN with 1 nearest neighbour and block size 7x7. The output will be saved as the path/to/KNNProcessed/neon_1_7.csv

##### To do (Updated on 12/07/15):
  * Use 1,300 images as training data and 700 as validation data;
  * Test all 2,000 images using the basic setting and generate confusion matrix as well as binary image files;
  * Zhiya - bs,resize,k = (9,0.1,1), (9,0.1,3), (9,0.1,5)
  * Meesha - bs,resize,k = (7,0.1,1),(7,0.1,3),(7,0.1,5)
  * Joel - bs,resize,k = (5,0.1,1),(5,0.1,3),(5,0.1,5)

##### Testing (Updated on Dec/13/2015):
 Execute from inside NeonKNN folder.
 
 Create greyscaleImages at relative path ../greyscaleImages: `mkdir ../greyscaleImages`
 
 `python main_skinPredicted.py <k> <block_size> <testIndexFrom> <testIndexTo>`
 
 Example: `python main_skinPredicted.py 9 5 0 2000` 
 
 will produce 2000 greyscale images at ../greyscaleImages for each of the test images (from index 0 to 1999)
