from flask import Flask, request, render_template
import data_handler
import threading
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


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

    print(file)

    dataset = data_handler.file_loader(file)
    stats = data_handler.map_all_stats(data_handler.retr_b_wave(dataset), 
                                       window_size)

    socketio.start_background_task(
        data_handler.send_over_UDP, stats, "127.0.0.1", port, delay, socketio
    )

    return render_template("display.html")


if __name__ == "__main__":
    socketio.run(app, debug=True)
