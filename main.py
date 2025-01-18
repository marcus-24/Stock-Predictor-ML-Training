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
from huggingface_hub import HfApi, login, Repository, snapshot_download

# local imports
from preprocessing.transformations import sequential_window_dataset
from models.lstm import build_model
from configs.mysettings import NeptuneSettings, HuggingFaceSettings
from configs.utils import set_env_name

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

# %% Define constant variables
DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
N_PAST = 10
N_FUTURE = 10
N_SHIFT = 1
EPOCHS = 1000
TRAIN_PERCENT = 0.75
N_FEATURES = len(DATA_COLS)

BRANCH_NAME: str = os.getenv("BRANCH_NAME")
ENV = set_env_name(BRANCH_NAME)

# %% Set up Neptune AI Tracking
neptune_settings = NeptuneSettings()
neptune_run = neptune.init_run(
    project="marcus-24/Neptune-Stock-Predictor",
    api_token=neptune_settings.NEPTUNE_TOKEN,
    tags=["test_model", "NOT_4_PRODUCTION"],
)  # track training metrics in neptune server

"""Save paramaters"""
neptune_run["parameters/batch_size"] = BATCH_SIZE
neptune_run["parameters/n_past"] = N_PAST
neptune_run["parameters/n_future"] = N_FUTURE
neptune_run["parameters/n_shift"] = N_SHIFT
neptune_run["parameters/tain_percent"] = TRAIN_PERCENT

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
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
df_train, df_val = train_test_split(df_train, test_size=0.5, random_state=42)

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
    train_set, n_past=N_PAST, n_features=N_FEATURES, batch_size=BATCH_SIZE
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
local_hugface_dir = "model_repo"
model_fname = os.path.join(local_hugface_dir, "model.h5")

hf_settings = HuggingFaceSettings()
login(hf_settings.HF_TOKEN)

repo_id = "DrMarcus24/test-stock-predictor"
if not os.path.exists(local_hugface_dir):
    repo = Repository(
        local_dir=local_hugface_dir,  # creates local repo for you
        clone_from=repo_id,  # path to remote repo
        token=True,  # use token from login function
    )
else:
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_hugface_dir)


# %%
"""Replace model with new one if MAE is better with new data"""
_, new_model_mae = model.evaluate(test_set)
existing_model = keras.models.load_model(model_fname)
_, existing_model_mae = existing_model.evaluate(
    test_set
)  # gather how existing model does with new data

if new_model_mae < existing_model_mae:  # only commit if new model is better

    model.save(model_fname)

    api = HfApi()

    # Upload the model file(s)
    api.upload_file(
        path_or_fileobj=model_fname,
        path_in_repo="model.h5",
        repo_id=repo_id,
    )

    print("New model has been uploaded to hugging face")
