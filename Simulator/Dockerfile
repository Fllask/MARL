#Deriving the latest base image
#FROM python:3.10
FROM nvcr.io/nvidia/pytorch:21.11-py3


WORKDIR C:\Users\valla\Documents\GitHub\MARL\Simulator
#COPY requirements.txt ./
COPY  . ./
RUN pip install -r ./requirements.txt
#to COPY the remote file at working directory in container

#ADD single_agent_gym.py
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

#CMD [ "python", "./single_agent_gym.py"]
