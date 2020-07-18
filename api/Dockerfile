# Start from the latest Long Term Support (LTS) Ubuntu version
FROM ubuntu:18.04

# Install pipenv
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY btc_predictor/ ./btc_predictor
COPY btc_predictor_api/ ./btc_predictor_api

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && pip3 install -r btc_predictor/requirements.txt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/btc_predictor/app.py