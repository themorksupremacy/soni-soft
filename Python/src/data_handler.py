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
from scipy.ndimage import gaussian_filter

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
    # 1. Separate the time column so it doesn't mess up the statistics
    time_col = dff['time']
    # Select only the frequency columns (all columns except 'time')
    spectral_data = dff.drop(columns=['time'])
    
    # 2. Compute moments across frequencies (axis=1) for every time point
    # This gives you a time series for each statistic
    stats_df = pd.DataFrame({
        "time": time_col,
        "mean": spectral_data.mean(axis=1),
        "std": spectral_data.std(axis=1),
        "skew": spectral_data.skew(axis=1),
        "kurtosis": spectral_data.kurtosis(axis=1)
    })

    # 3. Apply Rolling Statistics (as suggested by supervisor)
    # If you want to smooth the evolution over time, apply a rolling window
    # here. For example, a 10-step rolling mean:
    stats_df['mean_smooth'] = stats_df['mean'].rolling(window=10).mean()

    # 4. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_mtx = stats_df[["mean", "std", "skew", "kurtosis"]].corr()
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation of Spectral Moments Over Time")
    plt.savefig("heatmap.png")
    
    return stats_df

def band_stats(dff, freq_cols, window_size=10):
    time_col = dff['time']
    
    # Select band (multiple columns)
    band_data = dff[freq_cols]
    
    # Collapse band → single time series
    band_signal = band_data.mean(axis=1)
    
    # Rolling stats
    stats_df = pd.DataFrame({
        "time": time_col,
        "mean": band_signal.rolling(window_size).mean(),
        "std": band_signal.rolling(window_size).std(),
        "skew": band_signal.rolling(window_size).skew(),
        "kurtosis": band_signal.rolling(window_size).kurt()
    })
    
    return stats_df.dropna()
    

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
                kurtosis, range_min=0, range_max=1
            ),
        }
    )

# Mapping for frequency domain - patch 1
def map_all_stats_fdom1(data_frame, smoothing=False):

    window = int(data_frame["skew"].size * 0.15)

    if smoothing:
        skew = data_frame["skew"].rolling(window=window).mean()
        kurtosis = data_frame["kurtosis"].rolling(window=window).mean()
    else:
        skew = data_frame["skew"]
        kurtosis = data_frame["kurtosis"]

    return pd.DataFrame(
        {
            "mean": normalise_data(data_frame["mean"], range_min=1000, range_max=calc_max(data_frame["mean"], 1000)),
            "skew": map_skew_to_pan(data_frame["skew"]),
            "std": np.log1p(data_frame["std"]),
            "kurtosis": normalise_data(
                kurtosis, range_min=5, range_max=calc_max(data_frame["kurtosis"], 5)),
        }
    )

# Mapping for frequency domain - patch 2
def map_all_stats_fdom2(data_frame, smoothing=True):

    window = int(data_frame["skew"].size * 0.15)

    if smoothing:
        skew = data_frame["skew"].rolling(window=window).mean()
        kurtosis = data_frame["kurtosis"].rolling(window=window).mean()
    else:
        skew = data_frame["skew"]
        kurtosis = data_frame["kurtosis"]

    return pd.DataFrame(
        {
            "mean": normalise_data(data_frame["mean"], range_min=1000, range_max=calc_max(data_frame["mean"], 1000)),
            "skew": map_skew_to_pan(data_frame["skew"]),
            "std": np.log1p(data_frame["std"]),
            "kurtosis": normalise_data(
                kurtosis, range_min=5, range_max=calc_max(data_frame["kurtosis"], 5)
            ),
        }
    )

# Mapping for frequency domain - patch 3
def map_all_stats_fdom3(data_frame, smoothing=True):

    window = int(data_frame["skew"].size * 0.25)

    if smoothing:
        skew = data_frame["skew"].rolling(window=window).mean()
        kurtosis = data_frame["kurtosis"].rolling(window=window).mean()
    else:
        skew = data_frame["skew"]
        kurtosis = data_frame["kurtosis"]

    return pd.DataFrame(
        {
            "mean": normalise_data(data_frame["mean"], range_min=1000, range_max=calc_max(data_frame["mean"], 1000)),
            "skew": map_skew_to_pan(data_frame["skew"]),
            "std": np.log1p(data_frame["std"]),
            "kurtosis": normalise_data(
                kurtosis, range_min=20, range_max=calc_max(data_frame["kurtosis"], 20)
            ),
        }
    )

# Mapping for frequency domain - patch 4
def map_all_stats_fdom4(data_frame, smoothing=True):

    window = int(data_frame["skew"].size * 0.15)

    if smoothing:
        skew = data_frame["skew"].rolling(window=window).mean()
        kurtosis = data_frame["kurtosis"].rolling(window=window).mean()
    else:
        skew = data_frame["skew"]
        kurtosis = data_frame["kurtosis"]

    return pd.DataFrame(
        {
            "mean": normalise_data(data_frame["mean"], range_min=20, range_max=100),
            "skew": map_skew_to_pan(data_frame["skew"]),
            "std": np.log1p(data_frame["std"]),
            "kurtosis": normalise_data(
                kurtosis, range_min=5, range_max=calc_max(data_frame["kurtosis"], 5)
            ),
        }
    )

def map_skew_to_pan(skew_values):
    max_pos = max([x for x in skew_values if x > 0], default=0)
    max_neg = abs(min([x for x in skew_values if x < 0], default=0))

    pans = []
    for x in skew_values:
        if x > 0:
            pan = x / max_pos if max_pos != 0 else 0
        elif x < 0:
            pan = x / max_neg if max_neg != 0 else 0
        else:
            pan = 0
        pans.append(max(-1, min(1, pan)))  # clamp just in case

    return pans

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

def plot_power_spectrum_spectrogram(df, output_path="Python/src/Spectrogram.png"):
    df = df.copy()
    time = df.iloc[:, 0].values
    freqs = np.array([float(c) for c in df.columns[1:]])
    power = df.iloc[:, 1:].values

    power_db = 10 * np.log10(power + 1e-4)

    power_db = gaussian_filter(power_db, sigma=[0.5, 0.8])

    plt.figure(figsize=(12, 6))

    im = plt.pcolormesh(
        time,
        freqs,
        power_db.T,
        shading="gouraud",
        cmap="inferno" 
    )

    im.set_clim(vmin=np.percentile(power_db, 10), vmax=np.percentile(power_db, 99))

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Refined Power Spectrum")
    plt.colorbar(im, label="Power (dB)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return power_db

def plot_total_power(df, output_path="Python/src/TotalPower.png"):
    time = df.iloc[:, 0].values
    power = df.iloc[:, 1:].values

    total_power = np.sum(power, axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(time, total_power)

    plt.xlabel("Time [s]")
    plt.ylabel("Total Power")
    plt.title("Total Wave Power vs Time")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return total_power

def plot_peak_corr(file):
    df = pd.read_csv(file)
    
    # If there is only 1 peak, correlation is mathematically impossible
    if len(df) < 2:
        print("Not enough peaks to calculate correlation.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # Define the pairs to correlate
    pairs = [
        ('Std Dev (Peak)', 'Peak Std Dev'),
        ('Kurtosis (Peak)', 'Peak Kurtosis'),
        ('Avg Std Deviation (Dataset)', 'Dataset Avg Std Dev'),
        ('Avg Kurtosis (Dataset)', 'Dataset Avg Kurtosis')
    ]

    for i, (col, title) in enumerate(pairs):
        # Check if the column has variation (std > 0)
        if df[col].nunique() > 1:
            sns.regplot(ax=axes[i], data=df, x=col, y='Duration (Samples)', 
                        scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            corr_val = df['Duration (Samples)'].corr(df[col])
            axes[i].set_title(f'{title}\nCorr: {corr_val:.2f}')
        else:
            # Handle the case where all values are identical
            axes[i].scatter(df[col], df['Duration (Samples)'], alpha=0.5)
            axes[i].set_title(f'{title}\nCorr: N/A (No Variance)')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Duration (Samples)')

    plt.tight_layout()
    plt.savefig(r"Python\src\correlation.png")
    plt.close()


def get_ipi_heartbeat(dataframe, peak_details, base_delay_ms):
    """
    Calculates the Inter-Peak Interval.
    Returns an array where each index contains the 'tempo' set by the 
    most recent gap between two peaks.
    """
    ipi_values = np.zeros(len(dataframe))
    
    
    if len(peak_details) < 2:
        return ipi_values 

    current_interval = 0
    
    for i in range(1, len(peak_details)):
        prev_start = peak_details[i-1]['start_index']
        curr_start = peak_details[i]['start_index']
        
        
        interval_frames = curr_start - prev_start
        
       
        interval_ms = interval_frames * (base_delay_ms * 1000)
        
        
        next_peak_start = peak_details[i+1]['start_index'] if i+1 < len(peak_details) else len(dataframe)
        ipi_values[curr_start:next_peak_start] = interval_ms

    return ipi_values

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
