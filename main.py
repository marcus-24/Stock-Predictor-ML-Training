# %%
# standard imports
import yfinance as yf
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off one DNN custom operations
import keras
from keras import models
import tensorflow as tf
import warnings
import neptune
from neptune.types import File
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from neptune.integrations.tensorflow_keras import NeptuneCallback
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from datasets import Dataset
import matplotlib.pyplot as plt

# local imports
from preprocessing.transformations import (
    df_to_dataset,
    feature_engineering,
)
from models.builders import build_model
from configs.mysettings import NeptuneSettings, HuggingFaceSettings
from configs.branchsettings import set_env_name

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

# %% Define constant variables
BATCH_SIZE = 64
WIN_SIZE = 7
N_PAST = 1
N_FUTURE = 1  # TODO: Find out how Evidently AI handles multiple targets and predictions
N_SHIFT = 1
EPOCHS = 1000
TEST_SIZE = 0.25
VAL_SIZE = 0.5  # split in relation to test size
N_LABELS = 1  # only predicting close price

BRANCH_NAME: str = os.getenv("BRANCH_NAME")
ENV = set_env_name(BRANCH_NAME)

# TODO: Make these environment variables where models and data is pushed to a separate branches depending on enviornment
REPO_ID = "DrMarcus24/stock-predictor"
DATASET_URL = "DrMarcus24/stock-predictor-data"

# %% Set up Neptune AI Tracking
neptune_settings = NeptuneSettings()
neptune_run = neptune.init_run(
    project="marcus-24/Neptune-Stock-Predictor",
    api_token=neptune_settings.NEPTUNE_TOKEN.get_secret_value(),
    tags=["test_model", "NOT_4_PRODUCTION"],
)  # track training metrics in neptune server

"""Save paramaters"""
neptune_run["parameters/batch_size"] = BATCH_SIZE
neptune_run["parameters/n_past"] = N_PAST
neptune_run["parameters/n_future"] = N_FUTURE
neptune_run["parameters/n_shift"] = N_SHIFT
neptune_run["parameters/test_size"] = TEST_SIZE
neptune_run["parameters/val_size"] = VAL_SIZE

neptune_run["environment"] = ENV

# %% Preprocess data
# TODO: Need limits to keep dates during trading hours and non holidays
start_time = datetime.today() - relativedelta(days=90)

df = yf.Ticker("AAPL").history(interval="1h", start=start_time)
transformed_df = feature_engineering(df, win_size=WIN_SIZE, n_future=N_FUTURE)


"""Split Data"""
df_train, df_test = train_test_split(
    transformed_df, test_size=TEST_SIZE, random_state=42
)
df_train, df_val = train_test_split(df_train, test_size=VAL_SIZE, random_state=42)

neptune_run["training/train/data_start_date"] = df_train.index[0]
neptune_run["training/train/data_end_date"] = df_train.index[-1]
neptune_run["training/validation/data_start_date"] = df_val.index[0]
neptune_run["training/validation/data_end_date"] = df_val.index[-1]
neptune_run["training/test/data_start_date"] = df_test.index[0]
neptune_run["training/test/data_end_date"] = df_test.index[-1]

train_set = df_to_dataset(df_train, batch_size=BATCH_SIZE)
val_set = df_to_dataset(df_val, batch_size=BATCH_SIZE)
test_set = df_to_dataset(df_test, batch_size=BATCH_SIZE)

# %% Create model
n_features = transformed_df.shape[1] - N_LABELS  # to account for label column
model: models.Sequential = build_model(
    train_set,
    n_past=N_PAST,
    n_features=n_features,
    n_labels=N_LABELS,
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
whole_set = df_to_dataset(transformed_df, batch_size=BATCH_SIZE)
whole_features = whole_set.map(lambda x, _: x)  # get only the features
predictions = model.predict(whole_features)  # create predictions
predictions = tf.reshape(
    predictions, [-1]
).numpy()  # reshape into a single dimensional array

transformed_df["predictions"] = predictions

"""Save plot to Neptune"""
fig, ax = plt.subplots()
transformed_df[["label", "predictions"]].plot(ax=ax)
ax.grid(True)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Stock Price ($)")
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

# if True:
if not repo_exists or new_model_mae < existing_model_mae:

    """Create dataset for hugging face"""
    # do not want to save features to hugging face
    # TODO: Create feature store
    hugging_df = transformed_df[["predictions", "label"]].join(df, how="inner")

    """Push dataset and model to hugging face"""
    dataset = (
        Dataset.from_pandas(hugging_df)
        .train_test_split(test_size=TEST_SIZE)
        .push_to_hub(DATASET_URL)
    )
    model.save(model_url)
