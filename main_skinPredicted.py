import sys
from kNNskl_skinPredicted import NeonKNN_test

k = int(sys.argv[1])
block_size= int(sys.argv[2])
fromTestIndex = int(sys.argv[3])
toTestIndex = int(sys.argv[4])
fn = 'skinPredicted_k{}_bs{}_from-{}_to-{}.csv'.format(k, block_size,fromTestIndex,toTestIndex)

NeonKNN_test(k, block_size, fromTestIndex, toTestIndex, fn)

