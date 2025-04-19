# %%
# standard imports
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off one DNN custom operations
import keras
from keras import models
import hopsworks
import warnings
import neptune
from neptune.types import File
from sklearn.model_selection import train_test_split
from neptune.integrations.tensorflow_keras import NeptuneCallback
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup
import pandas as pd
import yfinance as yf

# local imports
from preprocessing.transformations import (
    df_to_dataset,
)
from preprocessing.predictions import format_predictions, performance_plot
from models.builders import build_model
from configs.mysettings import NeptuneSettings, HuggingFaceSettings, HopsworksSettings
from configs.branchsettings import set_env_name
from myfeatures.mlops import delete_existing_feature_group

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")

load_dotenv(override=True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

# %% Define constant variables
BATCH_SIZE = 32
EPOCHS = 1000
TEST_SIZE = 0.25
VAL_SIZE = 0.5  # split in relation to test size

BRANCH_NAME: str = os.getenv("BRANCH_NAME")
ENV = set_env_name(BRANCH_NAME)

# TODO: Make these environment variables where models and data is pushed to a separate branches depending on enviornment
REPO_ID = "DrMarcus24/stock-predictor"

# %% Get Data from Feature Store
hopsworks_settings = HopsworksSettings()
project = hopsworks.login(
    api_key_value=hopsworks_settings.HOPSWORKS_KEY.get_secret_value()
)
fs: FeatureStore = project.get_feature_store()

features_fg = fs.get_feature_group(name="stock_features")
labels_fg = fs.get_feature_group(name="stock_labels")


query = features_fg.select_all().join(
    labels_fg.select_all(), left_on=["date"], right_on=["date"], join_type="inner"
)

feature_view = fs.create_feature_view(
    name="model_training_data",
    query=query,
)

df, _ = feature_view.training_data(description="stock prediction with labels")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()


# %% Split data and transform into tensorflow dataset
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=VAL_SIZE, random_state=42)


train_set = df_to_dataset(df_train, batch_size=BATCH_SIZE)
val_set = df_to_dataset(df_val, batch_size=BATCH_SIZE)
test_set = df_to_dataset(df_test, batch_size=BATCH_SIZE)

# %% Set up Neptune AI Tracking
neptune_settings = NeptuneSettings()
neptune_run = neptune.init_run(
    project="marcus-24/Neptune-Stock-Predictor",
    api_token=neptune_settings.NEPTUNE_TOKEN.get_secret_value(),
    tags=["model", "PRODUCTION"],
)  # track training metrics in neptune server

"""Save paramaters"""
neptune_run["parameters/batch_size"] = BATCH_SIZE
neptune_run["parameters/test_size"] = TEST_SIZE
neptune_run["parameters/val_size"] = VAL_SIZE
neptune_run["environment"] = ENV

neptune_run["training/train/data_start_date"] = df_train.index[0]
neptune_run["training/train/data_end_date"] = df_train.index[-1]
neptune_run["training/validation/data_start_date"] = df_val.index[0]
neptune_run["training/validation/data_end_date"] = df_val.index[-1]
neptune_run["training/test/data_start_date"] = df_test.index[0]
neptune_run["training/test/data_end_date"] = df_test.index[-1]

# %% Create model
n_features = len([col for col in df.columns if "label" not in col])
n_labels = df.shape[1] - n_features
model: models.Sequential = build_model(
    train_set,
    n_features=n_features,
    n_labels=n_labels,
    batch_size=BATCH_SIZE,
)

# %% Train model
early_stopping = keras.callbacks.EarlyStopping(patience=30)

history = model.fit(
    train_set,
    epochs=EPOCHS,
    validation_data=val_set,
    callbacks=[
        early_stopping,
        NeptuneCallback(run=neptune_run, base_namespace="training"),
    ],
)


# %% Analyze Predictions

"""Create predictions for whole dataset"""
whole_set = df_to_dataset(df, batch_size=BATCH_SIZE)
whole_features = whole_set.map(lambda x, _: x)  # get only the features
predictions = model.predict(whole_features)  # create predictions
predictions_df = format_predictions(predictions, features=df)
predictions_df.to_csv("predictions.csv")


# """Save plot to Neptune"""
financial_data = yf.Ticker("AAPL").history(start=df.index[0], end=df.index[-1])
fig = performance_plot(predictions_df, financial_data)

neptune_run["visuals/prediction_vs_target"].upload(File.as_html(fig))


neptune_run.stop()
# %% Replace model with new one if MAE is better with new data
"""Login into hugging face"""
hf_settings = HuggingFaceSettings()
login(hf_settings.HF_TOKEN.get_secret_value())

hf_api = HfApi()
repo_exists = REPO_ID in [
    my_model.modelId for my_model in hf_api.list_models(author="DrMarcus24")
]

model_url = f"hf://{REPO_ID}"
if repo_exists:
    # evaluate production model on test set
    existing_model: models.Sequential = models.load_model(model_url)

    # gather how existing model does with new data
    _, existing_model_mae = existing_model.evaluate(test_set)

    _, new_model_mae = model.evaluate(test_set)  # Evaluate new model on test set

if not repo_exists or new_model_mae < existing_model_mae:

    print("Saving a new version of the stock prediction model")

    model.save(model_url)

    fg_name = "trained_model_predictions"
    delete_existing_feature_group(fs, fg_name=fg_name)
    fg_predictions: FeatureGroup = fs.create_feature_group(
        name=fg_name,
        description="stores predictions of a fully trained model for the features used during training",
        primary_key=["date"],
        event_time="date",
        online_enabled=False,
    )

    fg_predictions.insert(predictions_df)
