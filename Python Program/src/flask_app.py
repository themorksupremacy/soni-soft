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


def peaks_to_csv():
    global last_peaks
    if not last_peaks:
        return "No peak data available", 400

    
    peaks_df = pd.DataFrame(last_peaks)

    peaks_df.columns = [
        "Peak Number",
        "Start Index",
        "End Index",
        "Duration (Samples)",
        "Kurtosis (Peak)",
        "Std Dev (Peak)",
        "Avg Kurtosis (Dataset)",
        "Avg Std Deviation (Dataset)",
        "Duration (Seconds)",
    ]

    # Save to a temporary CSV file
    csv_path = r"Python\src\peak_analysis.csv"
    peaks_df.to_csv(csv_path, index=False)

    return csv_path


# Downloads zip file containing exported sonification and heatmap
@app.route("/download_spectrogram")
def download():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    zip_path = os.path.join(base_dir, "plots.zip")
    
    pure_data_path = os.path.abspath(os.path.join(base_dir, "..", "PureData", "sonification.wav"))
    heatmap_path = os.path.join(base_dir, "heatmap.png")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        if os.path.exists(r"Pure Data Patch\sonification.wav"):
            zipf.write(r"Pure Data Patch\sonification.wav", "sonification.wav")
        if os.path.exists(heatmap_path):
            zipf.write(heatmap_path, "heatmap.png")


    return send_file(
        zip_path, as_attachment=True, download_name=f"{base_name}.zip", max_age=0
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


def get_current_delay():
    return current_delay


# Stops the send_over_UDP method from sending data to PD and chartjs.
@socketio.on("stop_sonification")
def stop_sonification():
    stop_event.set()


# Configuration page
@app.route("/")
def index():
    return render_template("form.html")

def format_date(date):
    date = f"{date[:4]}/{date[4:6]}/{date[6:]}"
    return date



@app.route("/display", methods=["POST"])
def display():
    # Retrieve necessary values for sonification process
    file = request.files["dataset"]
    port = int(request.form["port"])
    patch_type = request.form.get("patch_type")

    print("Processing dataset as Satellite Powerspectrum data")

    global current_delay, b_wave, sampling_freq, base_name, normalised_stats, last_peaks, raw_stats
    current_delay = float(request.form["delay"])
    stop_event.clear()

    dataset = data_handler.file_loader(file)
    base_name = os.path.splitext(file.filename)[0]

    # 'Waveform_'+Date'_'+hour of measurement+'_'+BFI_Value 
    details = base_name.split('_')
    if len(details) >= 4:
        dataset_details = {
            "date": format_date(details[1]),
            "hour": details[2],
            "bfi": "0." + details[4],
        }
    else:
        dataset_details = {"date": "Unknown", "hour": "Unknown", "BFI": "Unknown"}

    raw_stats = data_handler.plot_spectral_parameters(dataset)

    if "skew" not in raw_stats.columns:
        raw_stats["skew"] = 0.0
        
    data_handler.plot_total_power(dataset)

    if patch_type == "patch1":
        normalised_stats = data_handler.map_all_stats_f1(raw_stats)

    def calculate_limits(df):
        limits = {}
        for col in ["mean", "skew", "std", "kurtosis"]:
            c_min = df[col].min()
            c_max = df[col].max()
            margin = (c_max - c_min) * 0.05
            if margin == 0:
                margin = 1  # Fallback for flat data

            limits[col] = {
                "min": round(float(c_min - margin), 2),
                "max": round(float(c_max + margin), 2),
            }
        return limits

    chart_limits = calculate_limits(raw_stats)
    peak_details = data_handler.get_peak_list(raw_stats, sensitivity=1.5)

    for peak in peak_details:
        peak["duration_sec"] = round(peak["duration"] * current_delay, 3)

    if peak_details:
        avg_duration = sum(p["duration"] for p in peak_details) / len(peak_details)
    else:
        avg_duration = 0

    print(f"Detected {len(peak_details)} peaks.")
    print(f"Peak List: {peak_details}")
    print(f"Average Duration: {avg_duration}")

    socketio.start_background_task(
        data_handler.send_over_UDP,
        normalised_stats,
        raw_stats,
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
        limits=chart_limits,
        file_info=dataset_details,
        filename=base_name
    )


if __name__ == "__main__":
    socketio.run(app, debug=True)