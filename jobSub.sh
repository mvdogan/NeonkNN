#!/bin/sh
module load python
## Name of job
#$-N test-1-9-1
## Concatenate SGE output? (recommend yes)
#$-j Y
## Parallel environment
#$-pe smp 1
## Next choose a queue
#$-q INFORMATICS
## Set the working directory to current directory
#$-cwd
## Set email address
#$-M zhiya-zuo@uiowa.edu
## Send me email at beginning, end, abort, suspend
#$-m beas
## Put your command or other application last
python main.py 1 9 neon_1_9.csv
