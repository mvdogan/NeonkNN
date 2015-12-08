import sys
from kNNskl import NeonKNN

k = int(sys.argv[1])
block_size= int(sys.argv[2])
fn = sys.argv[3]
NeonKNN(k, block_size, fn)

