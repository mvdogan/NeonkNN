#!/bin/sh
## Name of job
#$-N Santa-1-9-1
## Concatenate SGE output? (recommend yes)
#$-j Y
## Parallel environment
#$-pe smp 4
## Next choose a queue
#$-q AL
## Set the working directory to current directory
#$-cwd
## Set email address
#$-M joel-tosadojimenez@uiowa.edu
## Send me email at beginning, end, abort, suspend
#$-m beas
## Put your command or other application last
/Users/tosadojimenez/anaconda2/bin/python main_skinPredicted.py 9 5 0 10
