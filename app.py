from flask import Flask, jsonify, render_template, request
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
import base64
import numpy as np
import tempfile

from simulation import simulation

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/run_simulation')
def run_simulation():
    param1 = request.args.get('param1')
    param2 = request.args.get('param2')

    # Your simulation code here
    fig, ax = plt.subplots()
    x = [0]
    y = [0]
    line, = ax.plot(x, y)

    def update(frame):
        x.append(frame)
        y.append(frame)
        line.set_data(x, y)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(100), blit=True)

    # Save the animation to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.gif') as temp_file:
        ani.save(temp_file.name, writer='imagemagick')
        temp_file.seek(0)
        img_data = base64.b64encode(temp_file.read()).decode('utf-8')

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
