from flask import Flask, request, render_template
import data_handler
import threading
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
current_delay = 0.99

@socketio.on("update_delay")
def update_delay(data):
    global current_delay
    current_delay = float(data["delay"])
    print("New delay:", current_delay)

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/display", methods=["POST"])
def display():

    # Retrieve necessary values for sonification process
    file = request.files["dataset"]
    window_size = int(request.form["window_size"])
    port = int(request.form["port"])
    delay = float(request.form["delay"])

    dataset = data_handler.file_loader(file)

    # df = data_handler.retr_b_wave(dataset)
    # dff =data_handler.compute_stfft(df, 1024, 512, 35000)

    stats = data_handler.map_all_stats_tdom(data_handler.retr_b_wave(dataset), 
                                       window_size, magnitude=data_handler.get_mag(dataset))
    #stats = data_handler.map_all_stats_fdom(dff)

    socketio.start_background_task(
        data_handler.send_over_UDP, stats, "127.0.0.1", port, delay, socketio
    )

    return render_template("display.html")



if __name__ == "__main__":
    socketio.run(app, debug=True)
