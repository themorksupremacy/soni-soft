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

    if "time" in df.columns:
        return df

    # Return reference point to start of the file.
    file_name.seek(0)
    # Header removed since 'b_wave' is missing.
    df = pd.read_csv(file_name, header=None)

    # Give the simulated dataset the same column heading as satellite data.
    cols = ["b_wave"]
    df.columns = cols

    return df

def power_spectrum_file(dff):
    df = dff.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    stats_list = []

    for _, row in df.iterrows():
        row_series = row.dropna()
        if row_series.empty:
            continue
        stats_list.append({
            "mean": row_series.mean(),
            "skew": row_series.skew(),
            "std": row_series.std(),
            "kurtosis": row_series.kurtosis(),
        })

    Sxx_stats = pd.DataFrame(stats_list)

    Sxx_stats = Sxx_stats.fillna(0)

    f_heat = plt.figure()
    corr_mtx = Sxx_stats.corr(numeric_only=True)
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation heatmap")
    plt.savefig(r"Python\src\heatmap.png")
    plt.close(f_heat)

    return Sxx_stats
    

# Returns a dataseries of the 'b_wave' data retrieved from the csv dataset.
def retr_b_wave(data_frame):

    try:
        return data_frame["b_wave"]
    except Exception as e:
        print("Exception: ", e)
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
def calc_max(dataseries, min_val):
    range = dataseries.max() - dataseries.min()
    max = min_val + range
    return max

# Mapping for timeseries data.
def map_all_stats_tdom(data_series, window_size):
    x = data_series.rolling(window=window_size)

    # Normalise all other statistical moments to their respective ranges needed for the specified PD patch.
    return pd.DataFrame(
        {
            "mean": normalise_data(x.mean(), range_min=50, range_max=105),
            "skew": normalise_data(x.skew(), range_min=-1, range_max=1),
            "std": normalise_data(x.std(), range_min=50, range_max=127),
            "kurtosis": normalise_and_invert(x.kurt(), range_min=50, range_max=300),
        }
    )


# Mapping for frequency domain.
def map_all_stats_fdom(data_frame, smoothing=True):

    window = int(data_frame["skew"].size * 0.15)

    if smoothing:
        skew = data_frame["skew"].rolling(window=window).mean()
        kurtosis = data_frame["kurtosis"].rolling(window=window).mean()
    else:
        skew = data_frame["skew"]
        kurtosis = data_frame["kurtosis"]

    return pd.DataFrame(
        {
            "mean": normalise_data(data_frame["mean"], range_min=200, range_max=calc_max(data_frame["mean"], 200)),
            "skew": normalise_data(skew, range_min=0, range_max=1),
            "std": np.log1p(data_frame["std"]),
            "kurtosis": normalise_data(
                kurtosis, range_min=30, range_max=calc_max(kurtosis, 30)
            ),
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
    band = np.log10(Sxx_original[idx, :] + 1e-10)

    Sxx_stats = pd.DataFrame(
        {
            "mean": np.mean(band, axis=0),
            "skew": skew(band, axis=0, bias=False),
            "std": np.std(band, axis=0),
            "kurtosis": kurtosis(band, axis=0, bias=False),
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

# Calculates a threshold based on the mean of the Std Dev column
# Adjust 'sensitivity' to be higher if it's still picking up too much noise

def get_peak_list(dataframe, sensitivity=1):
    std_values = dataframe['std'].values
    
    threshold = std_values.mean() * sensitivity 
    
    is_peak = (std_values > threshold).astype(int)

    diff = np.diff(is_peak, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    peaks = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        duration = e - s

        # Calculate average stats during the peak period
        peak_kurt = dataframe['kurtosis'].iloc[s:e].mean()
        peak_std = dataframe['std'].iloc[s:e].mean()
        
        peaks.append({
            "id": i + 1,
            "start_index": int(s),
            "end_index": int(e),
            "duration": int(duration),
            "kurtosis": round(float(peak_kurt), 3),
            "std_dev": round(float(peak_std), 3),
            "kurt_avg": round(float(dataframe["kurtosis"].mean()), 3),
            "std_dev_avg": round(float(dataframe["std"].mean()), 3)
        })
        
    return peaks

def save_live_stats_plots(stats_df):
    """Generates static plots of the rolling stats for the ZIP download."""
    metrics = ['mean', 'skew', 'std', 'kurtosis']
    
    
    window_size = max(int(len(stats_df["skew"]) * 0.15), 1)

    for metric in metrics:
        # --- 1. Standard Plot (Always generated) ---
        plt.figure(figsize=(10, 4))
        plt.plot(stats_df[metric].values, color='royalblue', linewidth=2)
        plt.title(f"Recorded {metric.capitalize()} over Time")
        plt.grid(True, color='#ddd', linestyle='--')
        plt.tight_layout()
        
        # Saves as mean_chart.png, skew_chart.png, etc.
        plt.savefig(rf"Python\src\{metric}_chart.png")
        plt.close()

        # --- 2. Rolling Plot (Only for std and kurtosis) ---
        if metric in ['skew', 'kurtosis']:
            plt.figure(figsize=(10, 4))
            
            # Apply the 15% rolling window
            rolling_data = stats_df[metric].rolling(window=window_size, min_periods=1).mean()
            
            plt.plot(rolling_data.values, color='crimson', linewidth=2)
            plt.title(f"Smoothed {metric.capitalize()} (15% Rolling Window)")
            plt.grid(True, color='#ddd', linestyle='--')
            plt.tight_layout()
            
            # Saves as std_rolling_chart.png and kurtosis_rolling_chart.png
            plt.savefig(rf"Python\src\{metric}_rolling_chart.png")
            plt.close()

def plot_peak_corr(file):
    df = pd.read_csv(file)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5)) # Widened slightly for 4 plots

    sns.regplot(ax=axes[0], data=df, x='Std Dev (Peak)', y='Duration (Samples)', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    c_std = df['Duration (Samples)'].corr(df['Std Dev (Peak)'])
    axes[0].set_title(f'Peak Std Dev\nCorr: {c_std:.2f}')

    sns.regplot(ax=axes[1], data=df, x='Kurtosis (Peak)', y='Duration (Samples)', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    c_kurt = df['Duration (Samples)'].corr(df['Kurtosis (Peak)'])
    axes[1].set_title(f'Peak Kurtosis\nCorr: {c_kurt:.2f}')

    sns.regplot(ax=axes[2], data=df, x='Avg Std Deviation (Dataset)', y='Duration (Samples)', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    c_avg_std = df['Duration (Samples)'].corr(df['Avg Std Deviation (Dataset)'])
    axes[2].set_title(f'Dataset Avg Std Dev\nCorr: {c_avg_std:.2f}')

    sns.regplot(ax=axes[3], data=df, x='Avg Kurtosis (Dataset)', y='Duration (Samples)', 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    c_avg_kurt = df['Duration (Samples)'].corr(df['Avg Kurtosis (Dataset)'])
    axes[3].set_title(f'Dataset Avg Kurtosis\nCorr: {c_avg_kurt:.2f}')

    plt.savefig(r"Python\src\correlation.png")
    plt.close()


#Currently not working
def get_cumulative_peak_count(dataframe, peak_details, ceiling=100):
    counts = np.zeros(len(dataframe))
    
    for peak in peak_details:
        counts[peak['start_index']] = 1
        
    cumulative = np.cumsum(counts)
    
    inverted_counts = ceiling - (cumulative * 10)
    
    return np.maximum(inverted_counts, 0)

# ---UDP TRANSFER---

def send_over_UDP(
    dataframe,
    raw_stats,
    cumulative_peaks,
    host="127.0.0.1",
    port=8888,
    get_delay=None,
    socketio=None,
    stop_event=None,
):

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, skew, std, kurtosis, current_peaks, m, sk, st, k in zip(
            dataframe["mean"].to_numpy(),
            dataframe["skew"].to_numpy(),
            dataframe["std"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
            cumulative_peaks,
            raw_stats["mean"].to_numpy(),
            raw_stats["skew"].to_numpy(),
            raw_stats["std"].to_numpy(),
            raw_stats["kurtosis"].to_numpy(),

        ):

            # If the stop button has been clicked, data is no longer sent.
            if stop_event.is_set():
                print("Sonification stopped")
                break

            if np.isnan(mean):
                continue

            # Format data correctly for the Pure Data patch.
            msg = f"{mean} {skew} {std} {kurtosis} {current_peaks};\n"

            # Send the data via UDP to the port PD is listening on.
            s.sendto(msg.encode("utf-8"), (host, port))

            # Emit data to frontend for chartjs
            if socketio:
                socketio.emit(
                    "rolling_stats",
                    {
                        "mean": round(m, 2),
                        "skew": round(sk, 2),
                        "std": round(st, 2),
                        "kurtosis": round(k, 2),
                    },
                )

            # Get current delay from flask_app and sleep for the specified number of seconds/milliseconds.
            delay = get_delay()
            time.sleep(delay)
