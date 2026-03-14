# Imports
import pandas as pd
import numpy as np
import socket
import time
from scipy.signal import spectrogram
from scipy.stats import skew, kurtosis

# Code formatting rules:
# All data structures containing the statistical momements should be written in the same order everytime.
# This order being: mean, skew, standard deviation, kurtosis.
# Statistical moments will use full names as well (e.g. kurtosis, not kurt).


# Data Retrieval
def file_loader_sat(file_name):
    return 0


def file_loader_sim(file_name):
    df = pd.read_csv(file_name, header=None)
    df.columns = ["Original Data"]
    data_series = df["Original Data"]
    return data_series


def file_loader(file_name):

    df = pd.read_csv(file_name)

    df.columns = df.columns.str.lower()

    if "b_wave" in df.columns:
        return df
    
    #Since B_wave is missing the header can be removed so as to not lose the first row of data.
    file_name.seek(0)
    df = pd.read_csv(file_name, header=None)

    cols = ["b_wave"]
    df.columns = cols

    return df
    
def retr_b_wave(data_frame):

    try:
        return data_frame['b_wave']
    except Exception as e:
        print('Exception: ', e)
        return None

def get_mag(dataframe):
    if 'magnitude' in dataframe.columns:
        return dataframe['magnitude']
    else: 
        return None

def optimal_window():
    return 1

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
def map_all_stats_tdom(data_series, window_size, magnitude):
    x = data_series.rolling(window=window_size)

    if magnitude is None:
        magnitude = [0.5] * len(data_series)
    else:
        magnitude = [np.nan]*(window_size-1) + normalise_data(magnitude[window_size-1:], range_min=0.3, range_max=1)
    
    return pd.DataFrame(
        {
            "mean": normalise_data(x.mean(), range_min=50, range_max=105),
            "skew": normalise_data(x.skew(), range_min=-1, range_max=1),
            "std": normalise_data(x.std(), range_min=50, range_max=127),
            "kurtosis": normalise_and_invert(x.kurt(), range_min=50, range_max=300),
            "magnitude": magnitude,
        }
    )

def map_all_stats_fdom(data_frame):
    magnitude = [0.5] * len(data_frame["mean"])
    
    return pd.DataFrame(
    {
        "mean": normalise_data(data_frame["mean"], range_min=50, range_max=105),
        "skew": normalise_data(data_frame["skew"], range_min=-1, range_max=1),
        "std": normalise_data(data_frame["std"], range_min=50, range_max=127),
        "kurtosis": normalise_and_invert(data_frame["kurtosis"], range_min=50, range_max=127),
        "magnitude": magnitude,
    })


# Diagram Generation

def gen_Spectogram():
    return 0

def gen_freq_dom_comparison():
    return 0

# Fast Fourier Transforms (FFTs)
def compute_stfft(dataseries, nperseg_val, noverlap_val, sampling_freq):
    f_spec_original, t_spec_original, Sxx_original = spectrogram(
        dataseries, fs=sampling_freq, nperseg=nperseg_val, noverlap=noverlap_val
        )
    
    mean_vals = np.mean(Sxx_original, axis=1)
    skew_vals = skew(Sxx_original, axis=1)
    std_vals = np.std(Sxx_original, axis=1)
    kurt_vals = kurtosis(Sxx_original, axis=1)
    
    Sxx_stats = pd.DataFrame(
        {
            "mean": mean_vals,
            "skew": skew_vals,
            "std": std_vals,
            "kurtosis": kurt_vals,
        }
    )

    return Sxx_stats

# UDP Transfer
def send_over_UDP(dataframe, host="127.0.0.1", port=8888, get_delay=None, socketio=None, stop_event=None):
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, skew, std, kurtosis, magnitude in zip(
            dataframe["mean"].to_numpy(),
            dataframe["skew"].to_numpy(),
            dataframe["std"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
            dataframe["magnitude"].to_numpy(),
        ):
            
            if stop_event.is_set():
                print("Sonification stopped")
                break
            
            if np.isnan(mean):
                continue

            msg = f"{mean} {skew} {std} {kurtosis} {magnitude};\n"
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
                        "magnitude": round(magnitude, 2),
                    },
                )
            delay = get_delay()
            print(delay)
            time.sleep(delay)
