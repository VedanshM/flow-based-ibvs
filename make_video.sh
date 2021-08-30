#!/bin/bash
cd ./output
ffmpeg -f image2  -framerate 1 -i frame%05d.png -c:v libx264 -r 24   -y  out.mp4
