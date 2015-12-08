#!/bin/sh
## Name of job
#$-N skinDetection
## Concatenate SGE output? (recommend yes)
#$-j Y
## Parallel environment
#$-pe 16cpn 16
## Next choose a queue
#$-q all.q
## Set the working directory to current directory
#$-cwd
## Set email address
#$-M 
## Send me email at beginning, end, abort, suspend
#$-m beas
## Put your command or other application last
/Users/mvijayen/anaconda2/bin/python /Users/mvijayen/bda_project/NeonkNN/kNNskl2.py
