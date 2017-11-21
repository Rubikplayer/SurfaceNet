#!/bin/bash

# this script copy the ply files of the gipuma (iccv2015) to the DTU methods folder for evaluation

src_path="/home/mengqi/dataset/MVS/results/iccv2015"
dst_path="/home/mengqi/dataset/MVS/Points_MVS/gipuma"
mkdir $dst_path
for i in {1..120} #{1..5} / {0..10..2} / {1,3,5,2} / {003..008}
do
    src_file="${src_path}/dtu_accurate_$i/consistencyCheck-*/*.ply"
    dst_file=$(printf "${dst_path}/gipuma%03d_l3.ply" $i) # store results to variable
    #cp dtu_accurate_$i/consistencyCheck-*/*.ply $path/gipuma${i}_l3.ply
    cp $src_file $dst_file
    echo $i
done

