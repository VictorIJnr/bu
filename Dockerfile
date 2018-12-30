FROM python:3

WORKDIR /bu

#Copying everything will be temporary until I have fully setup distbu to work with JSON files
#Hopefully there's a GraphQL thing that allows JSON list pagination? Maybe. IDK...
#Okay I REALLY should get distbu up and running asap. This takes FOREVER.
COPY * /bu/

RUN pwd
RUN ls
RUN rm -rf Research/
RUN pip install -r src/requirements.txt
CMD ["bash"]