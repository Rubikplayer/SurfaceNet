#!/bin/bash

#block(name=3.2, threads=2, memory=8000, subtasks=1, gpu=true, hours=6)
    python -u test-3DdilatedConv.py # python will print out immediatly rather than store in buffer first.
    echo "Done" 
