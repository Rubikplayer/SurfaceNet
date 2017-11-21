#!/bin/bash

#block(name=4.2-reconstr, threads=1, memory=5000, subtasks=1, gpu=true, hours=100)
    python -u deepSurf.py # python will print out immediatly rather than store in buffer first.
    echo "Done" 

# #block(name=4.2-reconstr, threads=1, memory=5000, subtasks=1, gpu=true, hours=100)
# #block(name=3.2-train, threads=1, memory=15000, subtasks=1, gpu=true, hours=100)
