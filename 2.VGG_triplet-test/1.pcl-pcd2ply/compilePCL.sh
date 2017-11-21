#!/bin/bash

#block(name=2.1, threads=5, memory=28000, subtasks=1, gpu=true, hours=36)
    set -e # To exit the script as soon as one of the commands failed

    # http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php
    mkdir -p ~/libs && cd ~/libs
    wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.0.tar.gz # change it if needed
    tar xvf pcl-1.8.0.tar.gz

    cd pcl-pcl-1.8.0 && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j5 # not larger than the threads value of qsub

    echo "Done"

