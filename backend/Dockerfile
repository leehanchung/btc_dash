# pull official base image
FROM python:3.7-slim-buster 

# set work directory
WORKDIR /usr/src/backend

# set environment variables
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# set flask env
ENV FASTAPI_ENV development

# install system dependencies
RUN apt-get clean && apt-get update

# adding openssl flags and libs for building psycopg2
ENV LDFLAGS="-L/usr/local/opt/openssl/lib"
ENV CPPFLAGS="-I/usr/local/opt/openssl/include" 
RUN apt-get install -y python3-dev \
    && apt-get install -y libpq-dev \
    && apt-get -y install pipenv

RUN pip install --upgrade pip

# install dependencies using pipenv to align dev env
COPY Pipfile Pipfile.lock ./
RUN pipenv install --deploy --system

# copy source code to image
COPY . .

# run image
CMD [ "uvicorn", "app:app", "--host=0.0.0.0", "--port=5000", "--reload"]
