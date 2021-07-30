FROM public.ecr.aws/lambda/python:3.7

RUN python3.7 -m pip install --upgrade pip

COPY requirements.txt ./
RUN python3.7 -m pip install -r requirements.txt -t .

COPY btc_predictor ./btc_predictor
COPY api_serverless/app.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
