import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import socket
import time


def load_file(file):
    dataframe = pd.read_csv(file)
    return dataframe


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
        k = (num_kurtosis / (denominator * ((math.sqrt(sigma_sqr)) ** 4))) - 3
        kurtosis.append(k)

    results = pd.DataFrame(
        {"mean": means, "variances": variances, "kurtosis": kurtosis}
    )

    f_heat = plt.figure()
    corr_mtx = results.corr(numeric_only=True)
    sns.heatmap(corr_mtx, cmap="YlGnBu", annot=True)
    plt.title("Correlation heatmap")
    plt.savefig(r"Python\src\heatmap.png")
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
    fig.savefig("Parameter Plots")

    return results


def calc_max(dataseries, min_val):
    range = dataseries.max() - dataseries.min()
    max = min_val + range
    return max


def normalise(dataseries, target_min=0, target_max=1):
    col_min = dataseries.min()
    col_max = dataseries.max()

    if col_max == col_min:
        return dataseries * 0 + target_min

    return (dataseries - col_min) / (col_max - col_min) * (
        target_max - target_min
    ) + target_min


def send_all_over_UDP(dataframe, host="127.0.0.1", port=8888, delay=0.014627922):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for mean, variances, kurtosis in zip(
            dataframe["mean"].to_numpy(),
            dataframe["variances"].to_numpy(),
            dataframe["kurtosis"].to_numpy(),
        ):
            if np.isnan(mean):
                continue

            msg = f"{mean} {variances} {kurtosis};\n"
            s.sendto(msg.encode("utf-8"), (host, port))
            time.sleep(delay)


def plot_processed_results(dataframe, time_series, filename="Processed_Parameters.png"):

    aligned_time = time_series.iloc[dataframe.index]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle("Processed Spectral Parameters (Running Mean/Normalized)")

    ax1.plot(aligned_time, dataframe["mean"], "tab:blue")
    ax1.set_ylabel("Mean")

    ax2.plot(aligned_time, dataframe["variances"], "tab:orange")
    ax2.set_ylabel("Width (σ)")

    ax3.plot(aligned_time, dataframe["kurtosis"], "tab:purple")
    ax3.set_ylabel("Kurtosis (κ)")
    ax3.set_xlabel("Time")

    plt.tight_layout()
    fig.savefig(filename)
    print(f"Saved processed plot to {filename}")


def main():
    f1 = r"Python/src/Datasets/Satellite_Data/Whistler Wave Database/1st March 2013/PowerSpectrum_20130301_t02_0_03.csv"
    f2 = r"Python\src\Datasets\Satellite_Data\Whistler Wave Database\1st March 2013\PowerSpectrum_20130301_t02_0_16.csv"
    f3 = r"Python\src\Datasets\Satellite_Data\Whistler Wave Database\1st March 2013\PowerSpectrum_20130301_t02_0_73.csv"
    f4 = r"Python\src\Datasets\Satellite_Data\Whistler Wave Database\20th January 2016\PowerSpectrum_20160120_t19_0_04.csv"

    df = load_file(f4)
    results = plot_spectral_parameters(df)

    # ---OPTIONS---
    USE_RUNNING_MEAN = True
    WINDOW_SIZE = 8
    USE_NORMALIZATION = True
    KURTOSIS_THRESHOLD = 0.5

    final_data = results.copy()

    if USE_RUNNING_MEAN:
        # center=True ensures the average aligns with the middle of the window
        final_data = final_data.rolling(window=WINDOW_SIZE, center=True).mean().dropna()

    final_data["kurtosis"] = final_data["kurtosis"].apply(
        lambda x: 0 if abs(x) <= KURTOSIS_THRESHOLD else x
    )

    if USE_NORMALIZATION:
        final_data = pd.DataFrame(
            {
                "mean": normalise(final_data["mean"], 300, 5000),
                "variances": normalise(final_data["variances"], 60, 80),
                "kurtosis": normalise(
                    final_data["kurtosis"],
                    0,
                    1,
                ),
            }
        )

    plot_processed_results(final_data, df["time"])

    send_all_over_UDP(final_data)
    print("Done...!")


if __name__ == "__main__":
    main()
