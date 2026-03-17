from flask import Flask, request, render_template, send_file
import data_handler
import threading
from flask_socketio import SocketIO
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
current_delay = 0.99
stop_event = threading.Event()
b_wave = pd.Series()
sampling_freq = 35000
base_name = ""


@app.route("/download_spectrogram")
def download():
    return send_file(
        "Spectrogram.png", as_attachment=True, download_name=f"{base_name}.png"
    )


@socketio.on("update_delay")
def update_delay(data):
    global current_delay
    current_delay = float(data["delay"])
    print("New delay:", current_delay)


def get_current_delay():
    return current_delay


@socketio.on("stop_sonification")
def stop_sonification():
    stop_event.set()


@app.route("/")
def index():
    return render_template("form.html")


@app.route("/display", methods=["POST"])
def display():

    # Retrieve necessary values for sonification process
    file = request.files["dataset"]
    window_size = int(request.form["window_size"])
    port = int(request.form["port"])
    domain = request.form["domain"]

    global current_delay, b_wave, sampling_freq, base_name
    current_delay = float(request.form["delay"])

    stop_event.clear()

    dataset = data_handler.file_loader(file)
    base_name = os.path.splitext(file.filename)[0]

    b_wave = data_handler.retr_b_wave(dataset)

    if domain == "frequency":
        # dff = data_handler.compute_stfft(b_wave, 1024, 512, 35000)
        dff = data_handler.specific_freq_band(b_wave, 1024, 512, 35000)
        stats = data_handler.map_all_stats_fdom(dff)
    else:
        stats = data_handler.map_all_stats_tdom(
            data_handler.retr_b_wave(dataset),
            window_size,
            magnitude=data_handler.get_mag(dataset),
        )

    socketio.start_background_task(
        data_handler.send_over_UDP,
        stats,
        "127.0.0.1",
        port,
        get_current_delay,
        socketio,
        stop_event,
    )

    return render_template("display.html", delay=current_delay)


if __name__ == "__main__":
    socketio.run(app, debug=True)
