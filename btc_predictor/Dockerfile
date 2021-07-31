FROM public.ecr.aws/lambda/python:3.7 as baseimage

RUN python3.7 -m pip install --upgrade pip

COPY requirements.txt ./
RUN python3.7 -m pip install -r requirements.txt -t .

FROM baseimage as finalimage

COPY btc_predictor ./btc_predictor
COPY experiments ./experiments
COPY api/app.py ./

CMD ["app.lambda_handler"]
