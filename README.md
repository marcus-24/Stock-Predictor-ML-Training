# Stock-Predictor-ML-Training

## Objective

This repository is used to train the Tensorflow model, track the training metrics on Neptune AI, and deploy model artifacts to Hugging Face if the model accuracy has improved since the deployed model became stale.

## Project Setup

### Python Environment

First install the Python environment using:

`conda env create -f environment.yml`

Then you can activate the environment with the command below:
`conda activate stock_ml_train`

### Neptune AI Experiment Tracker

To track each training iteration, you will first need to create an account at <a href="https://neptune.ai/">Neptune.ai</a>. Then follow the <a href="https://docs.neptune.ai/setup/creating_project/">Create a Neptune Project</a> tutorial to create project that will store each experiment run.

In order for this code to communicate with Neptune.ai, you will need to provide it the API token as shown in the <a href="https://docs.neptune.ai/setup/setting_credentials/">Set Neptune credentials</a> page. This code requires the user to store the token as `NEPTUNE_TOKEN` in a `.env` file.

### Hugging Face Credentials

## Connected Services

This repository interacts with the following services below:

<ol>
<li><a href="https://github.com/marcus-24/Stock-Predictor-ML-Inference">Stock-Predictor-ML-Inference</a></li>
<li><a href="https://github.com/marcus-24/Stock-Predictor-Frontend">Stock-Predictor-Frontend</a></li>
</ol>
