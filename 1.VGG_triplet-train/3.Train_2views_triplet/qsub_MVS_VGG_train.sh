#!/bin/bash

#block(name=1.3, threads=2, memory=16000, subtasks=1, gpu=true, hours=26)
    python -u main_MVS_VGG.py # python will print out immediatly rather than store in buffer first.
    echo "Done" 
