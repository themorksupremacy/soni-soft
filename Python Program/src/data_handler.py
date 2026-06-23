# ---IMPORTS---
import pandas as pd
import numpy as np
import socket
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---DATA RETRIEVAL---


def file_loader(file_name):
    df = pd.read_csv(file_name)

    df.columns = df.columns.str.lower()

    if "b_wave" in df.columns:
        return df

    if "time" in df.columns:
        return df

    file_name.seek(0)
    df = pd.read_csv(file_name, header=None)

    cols = ["b_wave"]
    df.columns = cols

    return df

#Manual computation of spectral parameters using the equations specified by domain specialist
def plot_spectral_parameters(dataframe):
    means = []
    variances = []
    kurtosis = []

    df = dataframe.drop(columns=["time"])
    freqs = [float(f) for f in df.columns]
    freqs = np.array(freqs)

    for i, r in df.iterrows():
        magnitudes = r.values

        # ---Spectral Mean---
        numerator = sum(f * mag for f, mag in zip(freqs, magnitudes))
        denominator = sum(magnitudes)
        mean = numerator / denominator
        means.append(mean)

        # ---Spectral Width---
        numerator_width = sum(
            ((f - mean) ** 2) * mag for f, mag in zip(freqs, magnitudes)
        )
        sigma_sqr = numerator_width / sum(magnitudes)
        variances.append(sigma_sqr)

        # ---Kurtosis---
        num_kurtosis = sum(((f - mean) ** 4) * mag for f, mag in zip(freqs, magnitudes))
        k = (num_kurtosis / (denominator * ((np.sqrt(sigma_sqr)) ** 4))) - 3
        kurtosis.append(k)

    results = pd.DataFrame({"mean": means, "std": variances, "kurtosis": kurtosis})

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    heatmap_path = os.path.join(current_dir, "heatmap.png")

    f_heat = plt.figure()
    corr_mtx = results.corr(numeric_only=True)
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation heatmap")
    
    plt.savefig(heatmap_path)
    plt.close(f_heat)

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

    ax1.plot(dataframe["time"], means, "tab:blue")
    ax1.set_ylabel("Mean")

    ax2.plot(dataframe["time"], variances, "tab:orange")
    ax2.set_ylabel("Width (σ)")

    ax3.plot(dataframe["time"], kurtosis, "tab:purple")
    ax3.set_ylabel("Kurtosis (κ)")
    ax3.set_xlabel("Time")

    plt.tight_layout()
    plots_path = os.path.join(current_dir, "spectral_parameters.png")
    fig.savefig(plots_path)

    return results


# ---DATA NORMALISATION---


def normalise_data(dataseries, range_min, range_max):
    x_min = np.nanmin(dataseries)
    x_max = np.nanmax(dataseries)

    normalised_data = []

    for d in dataseries:
        t = (d - x_min) / (x_max - x_min)
        a = range_min + (t * (range_max - range_min))
        normalised_data.append(a)

    return normalised_data


# ---MAPPING FUNCTIONS---

def calc_max(dataseries, min_val):
    range_val = dataseries.max() - dataseries.min()
    max_val = min_val + range_val
    return max_val

#Mapping function for new spectral parameters with implemented kurtosis thresholding idea. Optimal threshold still yet to be found.
#Furthermore, a rolling average is applied to smooth erratic data
def map_all_stats_f1(data_frame, KURTOSIS_THRESHOLD=0.5, apply_smoothing=True, window_size=.05):
    if apply_smoothing:
        final_data = data_frame.rolling(
            window=int(data_frame["mean"].size * window_size), center=True
        ).mean().dropna()
    else:
        final_data = data_frame.copy()
    
    final_data["kurtosis"] = final_data["kurtosis"].apply(
        lambda x: 0 if abs(x) <= KURTOSIS_THRESHOLD else x
    )

    return pd.DataFrame(
        {
            "mean": normalise_data(final_data["mean"], 3000, 5000),
            "std": normalise_data(final_data["std"], 0, 1),
            "kurtosis": normalise_data(final_data["kurtosis"], 0, 1),
        }
    )


# ---PEAK DETECTION & ANALYSIS---


def get_peak_list(dataframe, sensitivity=1):
    std_values = dataframe["std"].values

    threshold = std_values.mean() * sensitivity

    is_peak = (std_values > threshold).astype(int)

    diff = np.diff(is_peak, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    peaks = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        duration = e - s

        # Calculate average stats during the peak period
        peak_kurt = dataframe["kurtosis"].iloc[s:e].mean()
        peak_std = dataframe["std"].iloc[s:e].mean()

        peaks.append(
            {
                "id": i + 1,
                "start_index": int(s),
                "end_index": int(e),
                "duration": int(duration),
                "kurtosis": round(float(peak_kurt), 3),
                "std_dev": round(float(peak_std), 3),
                "kurt_avg": round(float(dataframe["kurtosis"].mean()), 3),
                "std_dev_avg": round(float(dataframe["std"].mean()), 3),
            }
        )

    return peaks


def plot_total_power(df):
    time_vals = df.iloc[:, 0].values
    power = df.iloc[:, 1:].values

    total_power = np.sum(power, axis=1)

    plt.figure(figsize=(10, 4))
    plt.plot(time_vals, total_power)

    plt.xlabel("Time [s]")
    plt.ylabel("Total Power")
    plt.title("Total Wave Power vs Time")

    plt.tight_layout()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "TotalPower.png")

    plt.savefig(output_path)
    plt.close()

    return total_power


# ---UDP TRANSFER---


def send_over_UDP(
    dataframe,
    raw_stats,
    host="127.0.0.1",
    port=8888,
    get_delay=None,
    socketio=None,
    stop_event=None,
):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, std, kurtosis, m, st, k in zip(
            dataframe["mean"].to_numpy(),
            dataframe["std"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
            raw_stats["mean"].to_numpy(),
            raw_stats["std"].to_numpy(),
            raw_stats["kurtosis"].to_numpy(),
        ):
            
            if stop_event.is_set():
                print("Sonification stopped")
                break

            if np.isnan(mean):
                continue

            # Format data correctly for the Pure Data patch.
            msg = f"{mean} {std} {kurtosis};\n"

            s.sendto(msg.encode("utf-8"), (host, port))

            # Emit data to frontend for chartjs
            if socketio:
                socketio.emit(
                    "rolling_stats",
                    {
                        "mean": round(m, 2),
                        "std": round(st, 2),
                        "kurtosis": round(k, 2),
                    },
                )

            # Get current delay from flask_app and sleep
            delay = get_delay()
            time.sleep(delay)