# Imports
import pandas as pd
import numpy as np
import socket
import time


# Data Retrieval
def file_loader(file_name):
    df = pd.read_csv(file_name, header=None)
    df.columns = ["Original Data"]
    data_series = df["Original Data"]
    return data_series


# Data normalisation
def normalise_data(dataframe, range_min, range_max):
    x_min = np.nanmin(dataframe)
    x_max = np.nanmax(dataframe)

    normalised_data = []

    for d in dataframe:
        t = (d - x_min) / (x_max - x_min)
        a = range_min + (t * (range_max - range_min))
        normalised_data.append(a)

    return normalised_data


def normalise_and_invert(dataframe, range_min, range_max):
    x_min = np.nanmin(dataframe)
    x_max = np.nanmax(dataframe)

    normalised_data = []

    for d in dataframe:
        t = (d - x_min) / (x_max - x_min)
        inv_t = 1 - t
        a = range_min + (inv_t * (range_max - range_min))
        normalised_data.append(a)

    return normalised_data


# Mapping functions
def map_all_stats(data_series, window_size):
    x = data_series.rolling(window=window_size)

    return pd.DataFrame(
        {
            "mean": normalise_data(x.mean(), range_min=50, range_max=105),
            "skew": normalise_data(x.skew(), range_min=-1, range_max=1),
            "std": normalise_data(x.std(), range_min=50, range_max=127),
            "kurtosis": normalise_and_invert(x.kurt(), range_min=50, range_max=300),
        }
    )


# Fast Fourier Transforms (FFTs)


# UDP Transfer
def send_over_UDP(dataframe, host="127.0.0.1", port=8888, delay=0.1, socketio=None):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, skew, std, kurtosis in zip(
            dataframe["mean"].to_numpy(),
            dataframe["skew"].to_numpy(),
            dataframe["std"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
        ):
            if np.isnan(mean):
                continue

            msg = f"{mean} {skew} {std} {kurtosis};\n"
            s.sendto(msg.encode("utf-8"), (host, port))

            # Emit to frontend
            if socketio:
                socketio.emit(
                    "rolling_stats",
                    {
                        "mean": round(mean, 2),
                        "skew": round(skew, 2),
                        "std": round(std, 2),
                        "kurtosis": round(kurtosis, 2),
                    },
                )

            time.sleep(delay)
