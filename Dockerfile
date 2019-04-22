FROM python:3.6.8

WORKDIR /bu

#Copying everything will be temporary until I have fully setup distbu to work with JSON files
#Hopefully there's a GraphQL thing that allows JSON list pagination? Maybe. IDK...
#I'm no longer copying all of my data over, I'm using volumes instead. 
#Well until I get distbu fully-operational with all of it's GraphQL stuff at least.

#I'll need to use this with the relevant mounts on my laptop
#Need to explain that last comment a lil' bit more.
#THIS IS FINE. We use a bind mount for the src/ directory when running the container
COPY src/requirements.txt bu/src/

RUN pip install -r bu/src/requirements.txt
RUN python -m spacy download en

#I just really wanted less
RUN apt-get update
RUN apt-get install less
RUN cd bu/src

#This is the "exec" form, it's preferred but I like the "shell form" (the one I use) more
# ENTRYPOINT ["bash"]
ENTRYPOINT bash