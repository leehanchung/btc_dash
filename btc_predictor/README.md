# BTC Predictor

Machine learning code for BTC Predictor

# TODO:
- how to better architect the model version + data version?
    requirement: need to have model version tied to data version
    if we tied data version when initializing the model, we would need to initialize the model with the data to get both
    if we separate out data when initializing the model, then how do we get model version?
- fix lstm model training code so the basemodel framework runs.  the network code runs fine.
- add inference/predict code
- encapsulate the GARIMA training code into the basemodel framework