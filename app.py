from flask import Flask, jsonify, render_template, request
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
import base64
import numpy as np

from simulation import simulation

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_simulation")
def run_simulation():
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r', animated=True)

    def init():
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(-1.5, 1.5)
        return ln,

    def update(frame):
        xdata.append(frame)
        ydata.append(np.sin(frame))
        ln.set_data(xdata, ydata)
        return ln,


    ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 2 * np.pi, 128),
                                init_func=init, blit=True, interval=50)

    buf = io.BytesIO()
    ani.save(buf, format='gif')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/gif;base64,{img_data}" />'
    # hyperparameters = request.json
    # # Extract hyperparameters from the request
    # param1 = hyperparameters.get("param1")
    # param2 = hyperparameters.get("param2")
    # # Run the simulation with the provided hyperparameters
    # result = simulation(param1, param2)
    # return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
