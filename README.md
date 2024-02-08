<p align="center">
    <img style="width: 35%; height: 35%" src="quimia4ab.svg">
</p>
These are a bunch of files used for training models for the QUMIA project.


### Contents

These are the most important files:

* run-session.sh - the script that sets up a training session on the HPC environment: labelling, committing, copying and adding a train job to the job queue.

* train-job.sh / validate-job.sh - the SLURM jobs.

* train.py - the main script that sets up all the elements and performs the training and validation. It is actually based on an Python notebook (used for development purposes, but not committed due to potential data leaks) that is converted to train.py as part of the run-session.sh script.

* qumia_*.py - the actual code that is used by the train script, thematically spread over several modules.

* convert-notebook.sh - script that is used to convert Pyhon notebooks to stand-alone scripts.

* data-prep/ - directory with preprocessing code. These are also based on Python notebooks, but converted to scripts (see train.py).
