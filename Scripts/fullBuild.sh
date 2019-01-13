#!/bin/bash
#Builds the docker image and pushes it to Docker Hub

#Add a command line argument to pass as the tag
#If no tag presented, default to neg-1.0
docker image build -t victorijnr/bu:$1 ../

docker image push victorijnr/bu:$1