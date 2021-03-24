# RESOLVE_FOCAL_CONFLICTS

## ABOUT
# This script identifies calls that are currently not labelled as nonfocal, but 
# may actually be nonfocal. It is a wrapper for a pipeline of python scripts (jupyter
# notebooks, originally).

## DETAILED DESCRIPTION
# First, potential mislabelled calls are identified as calls that are very close
# in time and neither is labelled as nonfocal.These calls are pulled from 
# the dataset (01_identify_focal_conflicts.py) and identified to be the same call 
# or a different call based on comparison of spectrograms (02_assign_distance_score.py). 

# In same-call pairs or groups, the strongest call label remains focal and the 
# others are re-assigned to be nonfocal (Strength=intensity of the audio signal, 
# 03_resolve_conflicts.py).
# Lastly, the csv label files are updated to include the column "pred_focalType",
# which contains the updated labels (04_update_labelfiles.py).

## REQUIREMENTS
# Requirements are: 
# - the original labelfiles in EAS_shared/meerkat/working/processed/acoustic/total_synched_call_tables
# - call txts in EAS_shared/meerkat/working/processed/acoustic/extract_calls/txts/
# - a labelfile of ALL calls (labelfile.csv) in EAS_shared/meerkat/working/processed/acoustic/extract_calls/
#   (generated when generating txt call files)


## OUTPUT
# All output is saved in EAS_shared/meerkat/working/processed/acoustic/resolve_conflicts/

# - candidates_matches.json
# - candidates_labelfile.csv
# - f_nf.csv
# - pred_labelfile.csv
# - updated input csvs with ending '_resolved_conflicts.csv'

# All except the last one could be theoretically deleted but I would keep it, because it allows
# us to see what was done in the different steps of the pipeline.



# **********************************************************
# START HERE
# **********************************************************

# ***  1) Define server path   ***

# Indicate your path to EAS server
# (depends on how you mounted it and what OS you're working on)
# (e.g. "/Volumes" for Mac users, "\\10.126.19.90" on Windows etc..)

SERVER = "/Volumes"

# save information (will be accessed by scripts later)
write(SERVER, 'server_path.txt', sep="")


# ***  2) Setup python wrapper   ***

# Force reticulate to use the python version and environment 
# I uploaded on the server

# (Solution from https://github.com/rstudio/reticulate/issues/292)
ENV_PATH = os$path$join(os$path$sep, SERVER, 'EAS_shared',
                            'meerkat','working','processed',
                            'acoustic', 'resolve_conflicts', 'resolver_env', 'bin', 'python') 

Sys.setenv(RETICULATE_PYTHON = ENV_PATH)

library(reticulate)

# select environment
reticulate::use_condaenv("resolver_env", required = TRUE)
reticulate::py_config()


# ***  3) Run pipeline  ***

os = import("os")
PROJECT_HOME = os$path$join(os$path$sep, SERVER, 'EAS_shared',
                                     'meerkat','working','processed',
                                     'acoustic', 'resolve_conflicts')

# may wanna run one by one
py_run_file(os$path$join(os$path$sep, PROJECT_HOME, "01_identify_focal_conflicts.py"))
py_run_file(os$path$join(os$path$sep, PROJECT_HOME, "02_assign_distance_score.py"))
py_run_file(os$path$join(os$path$sep, PROJECT_HOME, "03_resolve_conflicts.py"))
py_run_file(os$path$join(os$path$sep, PROJECT_HOME, "04_update_labelfiles.py"))

# Clean up
file.remove('server_path.txt')

