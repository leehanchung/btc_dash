train:
	python train_model.py

profiler:
	python -m cProfile -o api.prof .\run_api.py

create-ecr-repo:
	repository="btc-predictor-function"
	account=$(aws sts get-caller-identity --query Account --output text)
	region="us-west-2"
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com
	aws ecr create-repository --repository-name $repository

create-s3-bucket:
	aws s3api create-bucket \
	--bucket btc-dash \
	--region us-west-2 \
	--create-bucket-configuration LocationConstraint=us-west-2

build:
	sam build
