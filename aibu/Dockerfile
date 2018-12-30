FROM python:3

WORKDIR /bu

#Copying everything will be temporary until I have fully setup distbu to work with JSON files
#Hopefully there's a GraphQL thing that allows JSON list pagination? Maybe. IDK...
COPY * /bu/

RUN pwd
RUN ls
RUN rm -rf /bu/Research
RUN pip install -r /bu/src/requirements.txt
CMD ["bash"]