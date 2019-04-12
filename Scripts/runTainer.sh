#!/bin/bash
#Runs a Docker container and gives it the name "bu"
#The name can be changed later on with command line args

#Needs a command line argument for the tag
docker container rm bu
clear
docker container run -it --name bu --mount src=distbu,dst=/bu/ \
    --mount type=bind,src=$(pwd)/src/,dst=/bu/src victorijnr/bu:$1

sudo chgrp docker /var/lib/docker/volumes/