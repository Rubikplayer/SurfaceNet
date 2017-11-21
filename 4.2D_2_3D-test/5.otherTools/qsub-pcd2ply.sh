#!/bin/bash

#block(name=4.3-pcd2ply, threads=1, memory=5000, subtasks=1, gpu=false, hours=24)
    #echo "process subtask $SUBTASK_ID of $N_SUBTASKS"
    python -u pcd2ply.py /home/mengqi/dataset/MVS/lasagne/save_reconstruction_result/adapt_thresh/model47-6viewPairs-resol0.400-strideRatio0.500/
    echo "Done" 
