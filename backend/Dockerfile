# pull official base image
FROM python:3.7-slim-buster 

# set work directory
WORKDIR /usr/src/backend

# set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV FLASK_ENV development

# install system dependencies
RUN apt-get update

# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# install dependencies using pipenv
# RUN pip install pipenv
# COPY Pipfile Pipfile.lock ./
# RUN pipenv install --deploy --system

# copy source code to image
COPY . .
EXPOSE 5000

# run image
# CMD [ "pipenv", "run", "flask", "run" ]
CMD [ "flask", "run"]
