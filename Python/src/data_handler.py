# Imports
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

    # Since B_wave is missing the header can be removed so as to not lose the first row of data.
    file_name.seek(0)
    df = pd.read_csv(file_name, header=None)

    cols = ["b_wave"]
    df.columns = cols

    return df


def retr_b_wave(data_frame):

    try:
        return data_frame["b_wave"]
    except Exception as e:
        print("Exception: ", e)
        return None


def get_mag(dataframe):
    if "magnitude" in dataframe.columns:
        return dataframe["magnitude"]
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
        magnitude = [np.nan] * (window_size - 1) + normalise_data(
            magnitude[window_size - 1 :], range_min=0.3, range_max=1
        )

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
            "kurtosis": normalise_data(
                data_frame["kurtosis"], range_min=50, range_max=127
            ),
            "magnitude": magnitude,
        }
    )


def audification(data_series, sampling_rate=35087):
    # Convert to numpy array immediately to use efficient memory handling
    data = data_series.to_numpy()

    x_min = data.min()
    x_max = data.max()

    range_val = x_max - x_min
    if range_val == 0:
        range_val = 1

    audio_data = (((data - x_min) / range_val) * 2 - 1) * 32767

    audio_data = audio_data.astype(np.int16)

    file_path = r"Python\src\data_audio.wav"
    wavfile.write(file_path, int(sampling_rate), audio_data)

    del audio_data

    return file_path


def specific_freq_band(dataseries, nperseg_val, noverlap_val, sampling_freq):

    f_spec_original, t_spec_original, Sxx_original = spectrogram(
        dataseries, fs=sampling_freq, nperseg=nperseg_val, noverlap=noverlap_val
    )

    f_low = 4000
    f_high = 5000

    idx = np.where((f_spec_original >= f_low) & (f_spec_original <= f_high))[0]

    band = Sxx_original[idx, :]

    Sxx_stats = pd.DataFrame(
        {
            "mean": np.mean(band, axis=0),
            "skew": skew(band, axis=0),
            "std": np.std(band, axis=0),
            "kurtosis": kurtosis(band, axis=0),
        }
    )

    # Spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(
        t_spec_original,
        f_spec_original,
        10 * np.log10(Sxx_original),
        shading="gouraud",
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.ylim(0, 5000)
    plt.colorbar(label="Intensity [dB]")
    plt.savefig(r"Python\src\Spectrogram.png")
    plt.close("all")

    # Correlation heatmap
    plt.figure()
    corr_mtx = Sxx_stats.corr()
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation heatmap")
    plt.savefig(r"Python\src\heatmap.png")
    plt.close("all")

    return Sxx_stats


def compute_playback(samples, col_num, sample_duration):
    dataset_duration = samples * sample_duration
    sampling_rate = 1 / sample_duration

    # Number of samples per column
    col_samples = samples / col_num
    playback = col_samples * sample_duration

    return playback


# Fast Fourier Transforms (FFTs)
def compute_stfft(dataseries, nperseg_val, noverlap_val, sampling_freq=35087.7):
    f_spec_original, t_spec_original, Sxx_original = spectrogram(
        dataseries, fs=sampling_freq, nperseg=nperseg_val, noverlap=noverlap_val
    )

    # mean_vals = np.mean(Sxx_original, axis=0)
    # skew_vals = skew(Sxx_original, axis=0)
    # std_vals = np.std(Sxx_original, axis=0)
    # kurt_vals = kurtosis(Sxx_original, axis=0)

    # Sxx_stats = pd.DataFrame(
    #     {
    #         "mean": mean_vals,
    #         "skew": skew_vals,
    #         "std": std_vals,
    #         "kurtosis": kurt_vals,
    #     }
    # )

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

    # Plot the spectrogram
    f_spec = plt.figure()
    plt.pcolormesh(
        t_spec_original, f_spec_original, 10 * np.log10(Sxx_original), shading="gouraud"
    )
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.title(f"Spectrogram (nperseg={nperseg_val}, noverlap={noverlap_val})")
    plt.ylim(
        0, 5000
    )  # Adjust y-axis limit based on expected frequency range of the signal
    plt.colorbar(label="Intensity [dB]")
    plt.savefig(r"Python\src\Spectrogram.png")
    plt.close(f_spec)

    return Sxx_stats


# UDP Transfer
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
