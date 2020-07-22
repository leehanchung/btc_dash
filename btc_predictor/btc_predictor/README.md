# BTC Predictor

Machine learning code for BTC Predictor

## Model API
Since we have models coming from multiple different frameworks, including Tensorflow and statsmodels, we have abstract away the basic model functions using a BaseModel API.

And every time we call model.fit, we will set the name of the model to `model` + `data` so we can have better understanding of how models are tied to the dataset for better versioning.

#### BaseModel API
To initialize, the API requires two parameter dictionarys. One for the model and one for the training.

BaseModel includes the following methods:

    fit():      fits model to the data that is loaded from DataReader class

    eval():     evaluate model from data loaded by DataReader class

    predict():  make prediction using the trained model

    load():     load an serialized and saved model
    
    save():     serialize and save the model

## Change Log
2020-07-18 - created a training script file `train.py`. training and evaluation code done. moved towards github issues framework.
