from flask import Flask, Response, render_template, request

from simulation import simulation

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_simulation")
def run_simulation():
    number_balls = int(request.args.get("param1"))
    max_velocity = float(request.args.get("param2"))
    return Response(simulation(number_balls, max_velocity), mimetype="image/gif")


if __name__ == "__main__":
    app.run(debug=True)
