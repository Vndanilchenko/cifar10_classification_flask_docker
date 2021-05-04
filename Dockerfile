# base image
FROM tiangolo/meinheld-gunicorn-flask

# update packages
RUN apt-get update

# copy files
COPY requirements.txt requirements.txt
COPY application application
COPY main.py main.py
# COPY cls cls

# install requirements
RUN python3 -m pip install -r requirements.txt
