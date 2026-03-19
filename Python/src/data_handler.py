# ---IMPORTS---
import pandas as pd
import numpy as np
import socket
import time
from scipy.signal import spectrogram
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

# Code formatting rules:
# All data structures containing the statistical momements should be written in the same order everytime.
# This order being: mean, skew, standard deviation, kurtosis.
# Statistical moments will use full names as well (e.g. kurtosis, not kurt).


# ---DATA RETRIEVAL---


def file_loader(file_name):

    df = pd.read_csv(file_name)

    # All column headings to lower case to avoid any errors due to case sensitivity.
    df.columns = df.columns.str.lower()

    # Stops if the file is already labelled / is not simulated data.
    if "b_wave" in df.columns:
        return df

    # Return reference point to start of the file.
    file_name.seek(0)
    # Header removed since 'b_wave' is missing.
    df = pd.read_csv(file_name, header=None)

    # Give the simulated dataset the same column heading as satellite data.
    cols = ["b_wave"]
    df.columns = cols

    return df


# Returns a dataseries of the 'b_wave' data retrieved from the csv dataset.
def retr_b_wave(data_frame):

    try:
        return data_frame["b_wave"]
    except Exception as e:
        print("Exception: ", e)
        return None


# Returns a dataseries of the magnitude data retried from the csv dataset.
def get_mag(dataframe):
    if "magnitude" in dataframe.columns:
        return dataframe["magnitude"]
    else:
        return None


# ---DATA NORMALISATION---


# Normalise a given dataseries to a specific range of values.
def normalise_data(dataseries, range_min, range_max):
    x_min = np.nanmin(dataseries)
    x_max = np.nanmax(dataseries)

    normalised_data = []

    for d in dataseries:
        t = (d - x_min) / (x_max - x_min)
        a = range_min + (t * (range_max - range_min))
        normalised_data.append(a)

    return normalised_data


# Normalise a given dataseries to a specific range of values and invert it.
# Useful for datastreams where sound output is not intuitive in PD.
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


# ---MAPPING FUNCTIONS---


# Mapping for timeseries data.
def map_all_stats_tdom(data_series, window_size, magnitude):
    x = data_series.rolling(window=window_size)

    # If magnitude is not present in the dataset, give it a consistent default value.
    # Magnitude scales the amplitude of the signal in PD.
    if magnitude is None:
        magnitude = [0.5] * len(data_series)
    else:
        # If magnitude is present, normalise it between 0 and 1.
        magnitude = [np.nan] * (window_size - 1) + normalise_data(
            magnitude[window_size - 1 :], range_min=0.3, range_max=1
        )

    # Normalise all other statistical moments to their respective ranges needed for the specified PD patch.
    return pd.DataFrame(
        {
            "mean": normalise_data(x.mean(), range_min=50, range_max=105),
            "skew": normalise_data(x.skew(), range_min=-1, range_max=1),
            "std": normalise_data(x.std(), range_min=50, range_max=127),
            "kurtosis": normalise_and_invert(x.kurt(), range_min=50, range_max=300),
            "magnitude": magnitude,
        }
    )


# Mapping for frequency domain.
def map_all_stats_fdom(data_frame):

    # As of right now magnitude is given a default value until I figure out how to compute this.
    magnitude = [0.5] * len(data_frame["mean"])

    return pd.DataFrame(
        {
            "mean": normalise_data(data_frame["mean"], range_min=50, range_max=105),
            "skew": normalise_data(data_frame["skew"], range_min=-1, range_max=1),
            "std": normalise_data(data_frame["std"], range_min=50, range_max=127),
            "kurtosis": normalise_data(
                data_frame["kurtosis"], range_min=50, range_max=127
            ),
            "magnitude": magnitude,
        }
    )


# --- AUDIFICATION---


def audification(data_series, sampling_rate=35087):
    # Convert pandas series to numpy
    data = data_series.to_numpy()

    x_min = data.min()
    x_max = data.max()

    range_val = x_max - x_min
    if range_val == 0:
        range_val = 1

    audio_data = (((data - x_min) / range_val) * 2 - 1) * 32767

    # Conversion to integer
    audio_data = audio_data.astype(np.int16)

    # Writing the WAV file
    file_path = r"Python\src\data_audio.wav"
    wavfile.write(file_path, int(sampling_rate), audio_data)

    del audio_data

    return file_path


# Compute the suggested playback rate based on the dataset and it's sampling rate.
def compute_playback(samples, col_num, sample_duration):
    dataset_duration = samples * sample_duration
    sampling_rate = 1 / sample_duration

    # Number of samples per column
    col_samples = samples / col_num
    playback = col_samples * sample_duration

    return playback


# ---FAST FOURIER TRANSFORMS (FFT)---


# FFT and return the data in a specified frequency range
def specific_freq_band(
    dataseries, nperseg_val, noverlap_val, sampling_freq, f_low, f_high
):
    f_spec_original, t_spec_original, Sxx_original = spectrogram(
        dataseries, fs=sampling_freq, nperseg=nperseg_val, noverlap=noverlap_val
    )

    # Slice the data to only include specified frequencies.
    idx = np.where((f_spec_original >= f_low) & (f_spec_original <= f_high))[0]

    f_focused = f_spec_original[idx]
    band = Sxx_original[idx, :]

    Sxx_stats = pd.DataFrame(
        {
            "mean": np.mean(band, axis=0),
            "skew": skew(band, axis=0),
            "std": np.std(band, axis=0),
            "kurtosis": kurtosis(band, axis=0),
        }
    )

    # Plot spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        t_spec_original,
        f_focused,
        10 * np.log10(band),
        shading="gouraud",
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")

    plt.ylim(f_low, f_high)

    plt.colorbar(label="Intensity [dB]")
    plt.title(f"Spectrogram ({f_low}Hz - {f_high}Hz)")
    plt.savefig(r"Python\src\Spectrogram.png")
    plt.close("all")

    # Plot heatmap
    f_heat = plt.figure()
    corr_mtx = Sxx_stats.corr(numeric_only=True)
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation heatmap")
    plt.savefig(r"Python\src\heatmap.png")
    plt.close(f_heat)

    return Sxx_stats


def compute_stfft(dataseries, nperseg_val, noverlap_val, sampling_freq=35087.7):
    f_spec_original, t_spec_original, Sxx_original = spectrogram(
        dataseries, fs=sampling_freq, nperseg=nperseg_val, noverlap=noverlap_val
    )

    Sxx_df = pd.DataFrame(Sxx_original)
    rows, cols = Sxx_original.shape
    print(rows)
    print(cols)

    stats_list = []

    for col in Sxx_df.columns:
        colSeries = Sxx_df[col]

        stats_list.append(
            {
                "mean": colSeries.mean(),
                "skew": colSeries.skew(),
                "std": colSeries.std(),
                "kurtosis": colSeries.kurtosis(),
            }
        )

    Sxx_stats = pd.DataFrame(stats_list)

    # Plot correlation heatmap
    f_heat = plt.figure()
    corr_mtx = Sxx_stats.corr(numeric_only=True)
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation heatmap")
    plt.savefig(r"Python\src\heatmap.png")
    plt.close(f_heat)

    # Plot spectrogram
    f_spec = plt.figure()
    plt.pcolormesh(
        t_spec_original, f_spec_original, 10 * np.log10(Sxx_original), shading="gouraud"
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title(f"Spectrogram (nperseg={nperseg_val}, noverlap={noverlap_val})")
    plt.ylim(0, 5000)
    plt.colorbar(label="Intensity [dB]")
    plt.savefig(r"Python\src\Spectrogram.png")
    plt.close(f_spec)

    return Sxx_stats


# ---UDP TRANSFER---


def send_over_UDP(
    dataframe,
    host="127.0.0.1",
    port=8888,
    get_delay=None,
    socketio=None,
    stop_event=None,
):

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, skew, std, kurtosis, magnitude in zip(
            dataframe["mean"].to_numpy(),
            dataframe["skew"].to_numpy(),
            dataframe["std"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
            dataframe["magnitude"].to_numpy(),
        ):

            # If the stop button has been clicked, data is no longer sent.
            if stop_event.is_set():
                print("Sonification stopped")
                break

            if np.isnan(mean):
                continue

            # Format data correctly for the Pure Data patch.
            msg = f"{mean} {skew} {std} {kurtosis} {magnitude};\n"

            # Send the data via UDP to the port PD is listening on.
            s.sendto(msg.encode("utf-8"), (host, port))

            # Emit data to frontend for chartjs
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

            # Get current delay from flask_app and sleep for the specified number of seconds/milliseconds.
            delay = get_delay()
            time.sleep(delay)
