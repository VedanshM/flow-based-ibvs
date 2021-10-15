#!/bin/bash
cd ./results
ffmpeg -f image2  -framerate 2 -i frame_%05d.png -c:v libx264 -r 24 -pix_fmt yuv420p  -y  out.mp4
