#!/bin/bash
#Builds the Docker image from scratch and then runs the container

#Stil need a tag as a command line argument
docker image build -t victorijnr/bu:$1 ../
docker image push victorijnr/bu:$1

docker container run -it --name bu --mount src=distbu,dst=/bu/ \ 
    --mount type=bind,src=$(pwd)/src,dst=/bu/src victorijnr/bu:$1