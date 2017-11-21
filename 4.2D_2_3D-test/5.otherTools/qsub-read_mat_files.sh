#!/bin/bash

#block(name=4.3-pcd2ply, threads=1, memory=5000, subtasks=1, gpu=false, hours=24)
    cat q.log/mat_files_adapth.txt | xargs -L 1 python read_from_single_mat.py
    echo "Done" 
#block(name=4.3-pcd2ply, threads=1, memory=5000, subtasks=1, gpu=false, hours=24)
    cat q.log/mat_files_fixth.txt | xargs -L 1 python read_from_single_mat.py
    echo "Done" 
