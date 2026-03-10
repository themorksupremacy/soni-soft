# Data handling imports
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os

# Data transfer imports
import socket
import time
import seaborn as sns


# ---Mapping Functions---
def map_to_freq(data, freq_min=200, freq_max=400):

    x_min = np.nanmin(data)
    x_max = np.nanmax(data)

    freq_data = []

    for d in data:
        t = (d - x_min) / (x_max - x_min)
        freq_data.append(freq_min + t * (freq_max - freq_min))

    return freq_data


def map_to_freq_invert(data, freq_min, freq_max):
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)

    freq_data = []

    for d in data:
        t = (d - x_min) / (x_max - x_min)
        inv_t = 1 - t
        freq_data.append(freq_min + inv_t * (freq_max - freq_min))

    return freq_data


def map_to_midi(data, midi_min=0, midi_max=127):
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)

    midi_data = []

    for d in data:
        t = (d - x_min) / (x_max - x_min)
        a = midi_min + (t * (midi_max - midi_min))
        midi_data.append(a)

    return midi_data


def normalise_skew(data, min=-1, max=1):
    x_min = np.nanmin(data)
    x_max = np.nanmax(data)

    skew_data = []

    for d in data:
        t = (d - x_min) / (x_max - x_min)
        a = min + (t * (max - min))
        skew_data.append(a)

    return skew_data


# ---Data calculations---
def running_mean(data_series, window_size):
    return data_series.rolling(window=window_size).mean()


def kurtosis(data_series, window_size):
    return data_series.rolling(window=window_size).kurt()


def skew(data_series, window_size):
    return data_series.rolling(window=window_size).skew()


def std_deviation(data_series, window_size):
    return data_series.rolling(window=window_size).std()


def compute_all_stats(data_series, window_size):
    x = data_series.rolling(window=window_size)

    return pd.DataFrame(
        {
            "mean": x.mean(),
            "kurtosis": x.kurt(),
            "skew": x.skew(),
            "std": x.std(),
        }
    )


def map_all_stats(data_frame, window_size):
    x = data_frame.rolling(window=window_size)

    return pd.DataFrame(
        {
            "mean": map_to_freq(data_frame["mean"], freq_min=200, freq_max=400),
            "std": map_to_midi(data_frame["std"]),
            "skew": data_frame["skew"],
            "kurt": map_to_freq(data_frame["kurtosis"], freq_min=1000, freq_max=1500),
        }
    )


def map_all_stats1(x):

    return pd.DataFrame(
        {
            "mean": map_to_freq(x.mean()),
            "std": map_to_freq_invert(x.std(), freq_min=250, freq_max=750),
            "skew": normalise_skew(x.skew()),
            "kurtosis": map_to_midi(x.kurt(), midi_min=0, midi_max=127),
        }
    )


# ---Data Transfer to PD---
def send_over_UDP(data, host="127.0.0.1", port=8888, delay=0.1):

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, port))
        for val in data:
            if np.isnan(val):
                continue
            msg = f"{float(val)}\n"
            s.sendall(msg.encode("utf-8"))
            time.sleep(delay)


def send_all_over_UDP(dataframe, host="127.0.0.1", port=8888, delay=0.1):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, std, skew, kurtosis in zip(
            dataframe["mean"].to_numpy(),
            dataframe["std"].to_numpy(),
            dataframe["skew"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
        ):
            if np.isnan(mean):
                continue

            msg = f"{mean} {std} {skew} {kurtosis};\n"
            s.sendto(msg.encode("utf-8"), (host, port))
            time.sleep(delay)


# ---Plotting Functions---
def animate_stats(
    data,
    window_size=1500,
    interval=50,
    host="127.0.0.1",
    port=8888,
    file_name="Animation.mp4",
):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 8))
    stats = ["mean", "std", "skew", "kurtosis"]
    lines = []

    df = pd.DataFrame(
        {
            "mean": map_to_midi(data["mean"], midi_min=50, midi_max=105),
            "std": map_to_midi(data["std"], midi_min=50, midi_max=127),
            "skew": normalise_skew(data["skew"]),
            "kurtosis": map_to_freq_invert(data["kurtosis"], freq_min=50, freq_max=300),
        }
    )

    for ax, stat in zip(axs, stats):

        clean = data[stat].dropna()

        y_min = clean.min()
        y_max = clean.max()
        padding = (y_max - y_min) * 0.1

        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_title(stat)
        ax.grid(True)

        (line,) = ax.plot([], [], linewidth=1.5)
        lines.append(line)

    def update(frame):

        start = max(0, frame - window_size)

        x_slice = data.index[start:frame]

        if len(x_slice) < 2:
            return lines

        for i, stat in enumerate(stats):
            y_slice = data[stat].iloc[start:frame]
            lines[i].set_data(x_slice, y_slice)

        axs[-1].set_xlim(x_slice.min(), x_slice.max())

        curr_mean = df["mean"].iloc[frame]
        curr_std = df["std"].iloc[frame]
        curr_skew = df["skew"].iloc[frame]
        curr_kurt = df["kurtosis"].iloc[frame]

        if np.isnan(curr_mean):
            return lines

        msg = f"{curr_mean} {curr_skew} {curr_std} {curr_kurt};\n"
        sock.sendto(msg.encode("utf-8"), (host, port))

        return lines

    ani = animation.FuncAnimation(
        fig, update, frames=range(window_size, len(data)), interval=interval, blit=False
    )

    # writer = FFMpegWriter(fps=30, bitrate=1800)
    # ani.save(file_name, writer=writer)
    # print(f"Animawtion recorded to {file_name}")

    plt.tight_layout()
    plt.show()

    sock.close()


def file_loader():
    file_name = r"Python\src\S_BFI_2_1_modified.csv"
    df = pd.read_csv(file_name, header=None)
    df.columns = ["Original Data"]
    data_series = df["Original Data"]
    return data_series


# def investigate_correlation(dataframe):
#     clean_df = dataframe.dropna()

#     corr_matrix = clean_df.corr()

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
#     plt.title("Correlation")
#     plt.show()

#     return corr_matrix


def main():

    val = file_loader()
    x = compute_all_stats(val, 1500)
    animate_stats(x, 1500, 2)


if __name__ == "__main__":
    main()
