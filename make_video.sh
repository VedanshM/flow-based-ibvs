#!/bin/bash

ffmpeg -f image2  -framerate 100 -i $1/results_dfvs/frame_%05d.png -c:v libx264 -r 24 -pix_fmt yuv420p  -y  $1/logs_dfvs/out.mp4
ln -sf $1/logs_dfvs/out.mp4 out.mp4
