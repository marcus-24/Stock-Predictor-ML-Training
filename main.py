# standard imports
import yfinance as yf
import keras
import os
import warnings
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# local imports
from preprocessing.transformations import sequential_window_dataset
from models.lstm import build_bidirec_lstm_model

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
N_PAST = 10
N_FUTURE = 10
N_SHIFT = 1
EPOCHS = 1000
TRAIN_PERCENT = 0.75
N_FEATURES = len(DATA_COLS)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    print("token recieved")
else:
    print("token was not passed")


df = yf.Ticker("AAPL").history(interval="1d", start="2016-01-01", end="2024-01-30")

"""Split Data"""
split_idx = int(TRAIN_PERCENT * df.shape[0])
split_time = df.index[split_idx]

x_train = df.loc[:split_time, DATA_COLS]
train_time = x_train.index.to_numpy()
x_val = df.loc[split_time:, DATA_COLS]
val_time = x_val.index.to_numpy()

train_set = sequential_window_dataset(
    x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
)
val_set = sequential_window_dataset(
    x_val, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
)

"""Create model"""
model = build_bidirec_lstm_model(
    train_set, n_past=N_PAST, n_features=N_FEATURES, batch_size=BATCH_SIZE
)

"""Train model"""
early_stopping = keras.callbacks.EarlyStopping(patience=10)

history = model.fit(
    train_set,
    epochs=EPOCHS,
    validation_data=val_set,
    callbacks=[early_stopping],
)

model_fname = os.path.join("data", "model.h5")
model.save(model_fname)


login(HF_TOKEN)

api = HfApi()
repo_id = "DrMarcus24/stock-predictor"
api.create_repo(repo_id=repo_id, exist_ok=True)

# Upload the model file(s)
api.upload_file(
    path_or_fileobj=model_fname,
    path_in_repo="model.h5",
    repo_id=repo_id,
)
