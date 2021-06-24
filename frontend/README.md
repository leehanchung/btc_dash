# BTC Predictor Web App Frontend


## Instructions
1. Create Python environment using pip or pipenv:

    ```
    pip install -r requirments.txt
    ```

    or

    ```
    pipenv install
    ```

2. Run the app:

    ```
    python run_dash.py
    ```
    or
    ```
    $Env:FLASK_APP="frontend/btc_dash:app"
    flask run
    ```
3. Open the app on your browser:

    ```
    localhost:5000
    ```

## Tech Stack
Python

Plotly Dash

Flask

## Local Testing
To run local testing, formatting, and linting, please do:

```
tox
```

## CI/CD
Testing, formatting, and linting pipeline is run on Github Action using `.github/workflows/frontend.yml`.

Currently the frontend is setup to be auto-deployed to Heroku for changes in the `dev` branch. Because we have a monorepo, we have to utilize [sub-directory Heroku Buildpack](https://github.com/timanovsky/subdir-heroku-buildpack) to redirect Heroku to build and deploy from `/frontend` directly. For more detailed instructions please see [this Stackoverflow question](https://stackoverflow.com/questions/39197334/automated-heroku-deploy-from-subfolder)
