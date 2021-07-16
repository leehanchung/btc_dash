FROM public.ecr.aws/lambda/python:3.7

COPY requirements.txt ./
RUN python3.7 -m pip install -r requirements.txt -t .

COPY btc_predictor ./btc_predictor
COPY app.py ./
COPY VERSION ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
