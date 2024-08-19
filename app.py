from flask import Flask, request, render_template, jsonify
from simulation import simulation

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/run_simulation', methods=['POST'])
# def run_simulation():
#     hyperparameters = request.json
#     # Extract hyperparameters from the request
#     param1 = hyperparameters.get('param1')
#     param2 = hyperparameters.get('param2')
#     # Run the simulation with the provided hyperparameters
#     result = simulation(param1, param2)
#     return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
