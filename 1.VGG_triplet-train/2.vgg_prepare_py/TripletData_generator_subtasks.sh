#!/bin/bash

#block(name=tripletData_py, threads=8, memory=6000, subtasks=128, hours=24)
    echo "This is a python program in the queue"
    #sleep 6000 # Makes the script pause for 10 seconds
    echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    #seq -f "%03g" 1 128 | parallel -j 12 --workdir $PWD python TripletData_generator_2.py {}
    python -u TripletData_generator_2.py $SUBTASK_ID
    echo "Done" 
