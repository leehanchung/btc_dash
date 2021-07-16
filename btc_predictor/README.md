# BTC Dash Machine Learning

# Predictor Design

To support multiple different sources of data and multiple different ways of modeling, we designed  two template classes: BaseDataset and BasePredictor.

We extend BaseDataset template for a particular data source and use start_time, end_time, and resolution to characterize the data.

Modeling can be done using any modeling library of choice.

We then uses the BasePredictor template to bridge models and data together. The BasePredictors have the basic `train`, `eval`, `predict`, `load`, and `save` methods.

# Deployment
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
[SAM Deploy](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-cli-command-reference-sam-deploy.html)