#!/bin/bash

echo $1
echo
python3 run_dfvs.py $1 && ./make_video.sh $1 && ./plot.py $1 && ./clear_res.sh $1
