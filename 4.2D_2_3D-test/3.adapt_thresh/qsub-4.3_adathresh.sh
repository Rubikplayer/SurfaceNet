#!/bin/bash

#block(name=4.3_adathresh, threads=2, memory=30000, subtasks=1, gpu=false, hours=100)
    #echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    python -u adapthresh.py
    echo "Done" 
