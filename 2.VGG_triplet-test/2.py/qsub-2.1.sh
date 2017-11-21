#!/bin/bash

#block(name=2.2, threads=2, memory=6000, subtasks=1, gpu=true, hours=24)
    #echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    python -u VGGface_MVS-ing.py
    echo "Done" 
