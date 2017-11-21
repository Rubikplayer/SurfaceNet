#!/bin/bash

#block(name=block-1, threads=3, memory=12000, subtasks=1, hours=24)
    seq -f "%03g" 0 130 | parallel -j 12 --workdir $PWD python selectLightIMG.py {}
