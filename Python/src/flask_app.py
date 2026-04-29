# ---IMPORTS---
from flask import Flask, request, render_template, send_file
import data_handler
import threading
from flask_socketio import SocketIO
import os
import matplotlib
import zipfile

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---GLOBAL VARIABLES---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
current_delay = 0.99
stop_event = threading.Event()
b_wave = pd.Series()
sampling_freq = 35087
base_name = ""
raw_stats = pd.DataFrame()
normalised_stats = pd.DataFrame()
port = 8888
last_peaks = []
cumulative_peak_array = []


# Used to play audification of the data in the browser
@app.route("/get_audification")
def get_audification():
    return send_file("data_audio.wav", mimetype="audio/wav", max_age=0)

def peaks_to_csv():
    global last_peaks
    if not last_peaks:
        return "No peak data available", 400

    # Convert the list of dictionaries to a DataFrame
    peaks_df = pd.DataFrame(last_peaks)
    
    # Rename columns for clarity in the CSV
    peaks_df.columns = [
        "Peak Number", "Start Index", "End Index", 
        "Duration (Samples)", "Kurtosis (Peak)", "Std Dev (Peak)", "Avg Kurtosis (Dataset)", "Avg Std Deviation (Dataset)", "Duration (Seconds)"
    ]

    # Save to a temporary CSV file
    csv_path = r"Python\src\peak_analysis.csv"
    peaks_df.to_csv(csv_path, index=False)

    return csv_path

# Downloads zip file including spectrogram, heatmap, and audification of the data.
@app.route("/download_spectrogram")
def download():

    plt.close('all')
    plt.clf()

    peaks = peaks_to_csv()

    data_handler.plot_peak_corr(peaks)

    data_handler.save_live_stats_plots(raw_stats)

    with zipfile.ZipFile(r"Python\src\plots.zip", "w") as zipf:
        zipf.write(r"Python\src\Spectrogram.png", "Spectrogram.png")
        zipf.write(r"Python\src\TotalPower.png", "TotalPower.png")
        zipf.write(r"Python\src\heatmap.png", "heatmap.png")
        zipf.write(r"Python\src\data_audio.wav", "audification.wav")
        zipf.write(rf"{peaks}", "peaks.csv")
        zipf.write(r"Python\src\correlation.png", "peak_correlation.png")
        for metric in ['mean', 'skew', 'std', 'kurtosis']:
            zipf.write(rf"Python\src\{metric}_chart.png", f"{metric}_live_history.png")
            if metric in ['skew', 'kurtosis']:
                zipf.write(rf"Python\src\{metric}_rolling_chart.png", f"{metric}_rolling_chart.png")

    return send_file(
        "plots.zip", as_attachment=True, download_name=f"{base_name}.zip", max_age=0
    )


# Starts playing the sonification again without having to refresh the page.
@socketio.on("start_sonification")
def start_sonification():
    global stop_event
    stop_event.clear()
    socketio.start_background_task(
        data_handler.send_over_UDP,
        normalised_stats,
        raw_stats,
        cumulative_peak_array,
        "127.0.0.1",
        port,
        get_current_delay,
        socketio,
        stop_event,
    )


# Used to update the delay/playback speed in realtime.
@socketio.on("update_delay")
def update_delay(data):
    global current_delay
    current_delay = float(data["delay"])
    print("New delay:", current_delay)


# Returns the global variable for the delay.
def get_current_delay():
    return current_delay


# Stops the send_over_UDP method from sending data to PD and chartjs.
@socketio.on("stop_sonification")
def stop_sonification():
    stop_event.set()


# Configuration page used to setup the sonification tool.
@app.route("/")
def index():
    return render_template("form.html")

# Display page for the sonification process.
@app.route("/display", methods=["POST"])
def display():

    # Retrieve necessary values for sonification process
    file = request.files["dataset"]
    port = int(request.form["port"])
    domain = request.form["domain"]
    patch_type = request.form.get("patch_type")
    dataset_type = request.form.get("dataset_type")

    freq_min = request.form.get("freq_min")
    freq_max = request.form.get("freq_max")

    if freq_min and freq_max:
        freq_min = float(freq_min)
        freq_max = float(freq_max)
    else:
        freq_min = None
        freq_max = None

    print(f"Processing {dataset_type} dataset in {domain} domain")

    global current_delay, b_wave, sampling_freq, base_name, normalised_stats, last_peaks, raw_stats, cumulative_peak_array
    current_delay = float(request.form["delay"])
    stop_event.clear()

    dataset = data_handler.file_loader(file)
    base_name = os.path.splitext(file.filename)[0]

    if dataset_type == "satellite_powerspectrum":
    # Already frequency-domain data
        raw_stats = data_handler.power_spectrum_file(dataset)
        #raw_stats = data_handler.band_stats(dataset, dataset.columns[15:17])
        data_handler.plot_power_spectrum_spectrogram(dataset)
        data_handler.plot_total_power(dataset)
        #audification = data_handler.audification(raw_stats, sampling_rate=sampling_freq)

    else:
        b_wave = data_handler.retr_b_wave(dataset)

        #audification = data_handler.audification(b_wave, sampling_rate=sampling_freq)
        current_delay = data_handler.compute_playback(b_wave.size, 407, 0.0000285)

        if domain == "frequency":
            if freq_min is not None:
                raw_stats = data_handler.specific_freq_band(
                    b_wave, 1024, 512, 35000, freq_min, freq_max
                )
            else:
                raw_stats = data_handler.compute_stfft(b_wave, 1024, 512)

        else:  
            window_size = int(request.form["window_size"])
            rolling = b_wave.rolling(window=window_size)
            raw_stats = pd.DataFrame({
                "mean": rolling.mean(),
                "skew": rolling.skew(),
                "std": rolling.std(),
                "kurtosis": rolling.kurt(),
            })

    if patch_type == "patch1":
        normalised_stats = data_handler.map_all_stats_fdom1(raw_stats)
    elif patch_type == "patch2":
        normalised_stats = data_handler.map_all_stats_fdom2(raw_stats)
    elif patch_type == "patch3":
        normalised_stats = data_handler.map_all_stats_fdom3(raw_stats)
    elif patch_type == "patch4":
        normalised_stats = data_handler.map_all_stats_fdom4(raw_stats)

    def calculate_limits(df):
        limits = {}
        for col in ['mean', 'skew', 'std', 'kurtosis']:
            
            c_min = df[col].min()
            c_max = df[col].max()
            margin = (c_max - c_min) * 0.05
            if margin == 0: margin = 1 # Fallback for flat data
            
            limits[col] = {
                "min": round(float(c_min - margin), 2),
                "max": round(float(c_max + margin), 2)
            }
        return limits

    chart_limits = calculate_limits(raw_stats)

    peak_details = data_handler.get_peak_list(raw_stats, sensitivity=1.5)
    
    
    for peak in peak_details:
        peak['duration_sec'] = round(peak['duration'] * current_delay, 3)

# Calculate average duration (samples)
    if peak_details:
        avg_duration = sum(p['duration'] for p in peak_details) / len(peak_details)
    else:
        avg_duration = 0

    print(f"Detected {len(peak_details)} peaks.")
    print(f"Peak List: {peak_details}")
    print(f"Average Duration: {avg_duration}")

   
    cumulative_peak_array = data_handler.get_ipi_heartbeat(raw_stats, peak_details, current_delay)

    socketio.start_background_task(
        data_handler.send_over_UDP,
        normalised_stats,
        raw_stats,
        cumulative_peak_array, 
        "127.0.0.1",
        port,
        get_current_delay,
        socketio,
        stop_event,
    )

    last_peaks = peak_details

    return render_template(
    "display.html", 
    delay=current_delay, 
    peak_details=peak_details, 
    avg_duration=round(avg_duration, 2),
    limits=chart_limits
)


if __name__ == "__main__":
    socketio.run(app, debug=True)
