import sys
from kNNskl import NeonKNN

k = int(sys.argv[1])
block_size= int(sys.argv[2])

fn = 'neon_{}_{}.csv'.format(k, block_size)

NeonKNN(k, block_size, fn)

