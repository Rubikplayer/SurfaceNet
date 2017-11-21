#!/bin/bash

#block(name=2.1, threads=5, memory=28000, subtasks=1, gpu=true, hours=36)
    set -e # To exit the script as soon as one of the commands failed

    # http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php
    mkdir -p ~/libs && cd ~/libs
    git clone https://github.com/PointCloudLibrary/pcl pcl-trunk

    cd pcl-trunk && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    make -j5 # not larger than the threads value of qsub

    echo "Done"

