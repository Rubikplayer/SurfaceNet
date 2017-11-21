#!/bin/bash

#block(name=4.3_fixthresh, threads=2, memory=40000, subtasks=1, gpu=false, hours=100)
    #echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    python -u fixthresh.py
    echo "Done" 
