#!/bin/bash
cd ./results
ffmpeg -f image2  -framerate 60 -i frame_%05d.png -c:v libx264 -r 24   -y  out.mp4
