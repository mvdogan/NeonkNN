#!/bin/sh
## Concatenate SGE output? (recommend yes)
#$-j Y
## Parallel environment
#$-pe smp 4
## Next choose a queue
#$-q INFORMATICS
## Set the working directory to current directory
#$-cwd
## Set email address
#$-M zhiya-zuo@uiowa.edu
## Send me email at beginning, end, abort, suspend
#$-m beas
## Name of job
#$-N test
## Put your command or other application last
module load python
python main_skinPredicted.py 9 5 0 10

