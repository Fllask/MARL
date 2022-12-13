#Deriving the latest base image
FROM python:3.10



WORKDIR C:\Users\valla\Documents\GitHub\MARL\dockertest
#COPY requirements.txt ./
COPY  . ./
RUN pip install -r ./requirements.txt
#to COPY the remote file at working directory in container

#ADD single_agent_gym.py
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

CMD [ "python", "./single_agent_gym.py"]
