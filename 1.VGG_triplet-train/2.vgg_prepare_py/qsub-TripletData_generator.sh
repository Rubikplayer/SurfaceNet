#!/bin/bash

#block(name=block-1, threads=8, memory=6000, subtasks=1, hours=24)
    echo "This is a python program in the queue"
    #sleep 6000 # Makes the script pause for 10 seconds
    seq -f "%03g" 1 128 | parallel -j 12 --workdir $PWD python TripletData_generator_2.py {}
    echo "Done" 
