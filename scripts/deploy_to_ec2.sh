#!/bin/bash

echo 'Deployment start...'
ssh ubuntu@127.0.0.1 "sudo docker image prune -f
	cd btc_dash
	sudo docker-compose down
	git fetch origin
	git reset --hard origin/staging
	sudo docker-compose build && sudo docker-compose up -d
	"

echo 'Deployment completed...'
