# BTC Dash Machine Learning

We deploy BTC Predictor machine learning model to AWS Lambda via AWS SAM CLI.

To deploy the Lambda function using SAM CLI:
```bash
# 1. Create ECR bucket for AWS SAM CLI to push images to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ${AWS ACCOUNT NUMBER}.dkr.ecr.us-west-2.amazonaws.com
aws ecr create-repository --repository-name btc-predictor-function

# 2. Create S3 bucket.
aws s3api create-bucket \
--bucket btc-predictor-deployment-bucket \
--region us-west-2 \
--create-bucket-configuration LocationConstraint=us-west-2

# 3. Build the Docker image from Dockerfile
sam build

# 4. To start the function locally, do
sam local start-api

# 5. And the API can be accessed by
curl http://localhost:3000/

# 6. Deploy the application to AWS Lambda
sam deploy

# 7. To remove the Lambda application
aws cloudformation delete-stack --stack-name Btc-Dash-BTCPredictorFunctionAndApi
```

# Reference
[Cookiecutter AWS SAM Python](https://github.com/aws-samples/cookiecutter-aws-sam-python/tree/master/%7B%7B%20cookiecutter.project_name%20%7D%7D)
[SAM Deploy](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-cli-command-reference-sam-deploy.html)