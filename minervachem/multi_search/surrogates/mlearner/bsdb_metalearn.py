import sys
import numpy as np
import pandas as pd

from metalearner import MetaLearner
from md_utilities import *

# take in current size (max substructure size) and min_points (to subselect a dataset)
csize = int(sys.argv[1])
min_points = int(sys.argv[2])

if min_points == 20:
    nsr = (5, 10, 15)
elif min_points == 50:
    nsr = (5, 10, 15, 20, 25, 30, 35, 40)
elif min_points == 100:
    nsr = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
else:
    print('The requested number of points is not supported.')
    raise SystemExit("Stopping execution here")

#invoke mlearner
mlearner = MetaLearner(f'databases/bigsoldb_minns_{min_points}_temp_298_15_nodubs_molarity.csv',
                       working_directory=None,
                       max_subgraph_size=csize, make_log=True)
# select cross validation range for alpha
mlearner.alpha_range = 'a2'
# run models 
mlearner.scan_monolayers(rnotes=f'bsdb_{min_points}_fulltaskspoints', res_file_name=f'onalltasks_bsdb_{min_points}_c10seeds_{csize}size',
                         task_range=tuple(mlearner.dataset.tasks), #not setting old tasks - should revert to using all that are not target task
                         mss_range=(csize,), 
                         nshots_range=nsr, 
                         rss_range=list(range(0, 10, 1)))
