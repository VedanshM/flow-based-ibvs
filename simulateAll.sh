#!/bin/bash

date_tag=$(date +%y%m%d_%H%M%S)
tag="${1:-$date_tag}"
echo $tag

mkdir -p saved/$tag

./simulate.sh ./data/arkansaw saved/$tag/arkansaw
./simulate.sh ./data/eudora saved/$tag/eudora
./simulate.sh ./data/pablo saved/$tag/pablo
./simulate.sh ./data/stokes saved/$tag/stokes
./simulate.sh ./data/hillsdale saved/$tag/hillsdale
./simulate.sh ./data/denmark saved/$tag/denmark
./simulate.sh ./data/quantico saved/$tag/quantico
./simulate.sh ./data/mesic saved/$tag/mesic
