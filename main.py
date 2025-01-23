# %%
# standard imports
import yfinance as yf
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off one DNN custom operations
import keras
import warnings
import neptune
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from neptune.integrations.tensorflow_keras import NeptuneCallback
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset
from requests.exceptions import HTTPError

# local imports
from preprocessing.transformations import sequential_window_dataset
from models.lstm import build_model
from configs.mysettings import NeptuneSettings, HuggingFaceSettings
from configs.utils import set_env_name

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")

load_dotenv(".env")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

# %% Define constant variables
DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
N_PAST = 10
N_FUTURE = 10
N_SHIFT = 1
EPOCHS = 2
TEST_SIZE = 0.25
VAL_SIZE = 0.5  # split in relation to test size
N_FEATURES = len(DATA_COLS)
N_LABELS = 1  # only predicting close price

BRANCH_NAME: str = os.getenv("BRANCH_NAME")
ENV = set_env_name(BRANCH_NAME)

# TODO: Make these environment variables where models and data is pushed to a separate branches depending on enviornment
MODEL_URL = "hf://DrMarcus24/test-stock-predictor"
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
end_date = date.today()
start_date = end_date - relativedelta(years=6)

df = (
    yf.Ticker("AAPL")
    .history(interval="1d", start=start_date, end=end_date)
    .loc[:, DATA_COLS]
)

"""Split Data"""
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=VAL_SIZE, random_state=42)

neptune_run["training/train/data_start_date"] = df_train.index[0]
neptune_run["training/train/data_end_date"] = df_train.index[-1]
neptune_run["training/validation/data_start_date"] = df_val.index[0]
neptune_run["training/validation/data_end_date"] = df_val.index[-1]
neptune_run["training/test/data_start_date"] = df_test.index[0]
neptune_run["training/test/data_end_date"] = df_test.index[-1]

train_set = sequential_window_dataset(
    df_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
)
val_set = sequential_window_dataset(
    df_val, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
)
test_set = sequential_window_dataset(
    df_test, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
)


# %% Create model
model = build_model(
    train_set,
    n_past=N_PAST,
    n_features=N_FEATURES,
    n_labels=N_LABELS,
    batch_size=BATCH_SIZE,
)

# %% Train model
early_stopping = keras.callbacks.EarlyStopping(patience=10)

history = model.fit(
    train_set,
    epochs=EPOCHS,
    validation_data=val_set,
    callbacks=[
        early_stopping,
        NeptuneCallback(run=neptune_run, base_namespace="training"),
    ],
)


neptune_run.stop()
# %% clone repo if needed
hf_settings = HuggingFaceSettings()
login(hf_settings.HF_TOKEN.get_secret_value())


# %% Replace model with new one if MAE is better with new data

try:
    _, new_model_mae = model.evaluate(test_set)  # Evaluate new model on test set
    existing_model = keras.models.load_model(
        MODEL_URL
    )  # evalute production model on test ste
    _, existing_model_mae = existing_model.evaluate(
        test_set
    )  # gather how existing model does with new data

    if new_model_mae < existing_model_mae:  # only commit if new model is better
        # TODO: not DRY
        dataset = (
            Dataset.from_pandas(df)
            .train_test_split(test_size=TEST_SIZE)
            .push_to_hub(DATASET_URL)
        )
        model.save(MODEL_URL)

except HTTPError as err:
    print("An error occurred: ", err)
    dataset = (
        Dataset.from_pandas(df)
        .train_test_split(test_size=TEST_SIZE)
        .push_to_hub(DATASET_URL)
    )
    model.save(MODEL_URL)
