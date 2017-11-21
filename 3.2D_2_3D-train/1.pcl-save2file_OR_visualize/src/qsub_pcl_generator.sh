#!/bin/bash

#block(name=block-1, threads=8, memory=10000, subtasks=1, hours=24)
    seq -f "%03g" 1 128 | parallel -j 8 --workdir $PWD ../build/matrix_transform {}
    echo "Done" 
