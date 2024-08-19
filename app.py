from flask import Flask, jsonify, render_template, request
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import io
import base64
import numpy as np
import tempfile
from simulation import Ball, check_collision, resolve_collision, build_balls, build_plot, IMAGES, image_pathroot
from typing import List
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

# from simulation import simulation

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/run_simulation')
def run_simulation():
    number_balls = request.args.get('param1')
    max_velocity = request.args.get('param2')

    # hyperparameters = request.json
    # # Extract hyperparameters from the request
    # param1 = hyperparameters.get("param1")
    # param2 = hyperparameters.get("param2")
    # # Run the simulation with the provided hyperparameters
    # result = simulation(param1, param2)
    # return jsonify(result)

    def animate(balls: List[Ball], ax, dt, bounds, centre_bounds):
        for ball in balls:
            ball.update_position(dt, bounds)

        for j in range(len(balls)):
            for k in range(j + 1, len(balls)):
                if check_collision(balls[j], balls[k]):
                    resolve_collision(balls[j], balls[k], centre_bounds)

        artists = []
        for ball in balls:
            artists.append(ball.draw(ax))

        # Define font style
        font = FontProperties()
        font.set_family("copperplate")
        font.set_size(30)
        font.set_weight("bold")

        # Check if all balls are the same kind
        kinds = [ball.kind for ball in balls]
        if len(set(kinds)) == 1:
            winner = kinds[0]
            ax.text(
                0.5,
                0.5,
                f"{winner.capitalize()} wins!\n\n\n",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="black",
                fontproperties=font,
                bbox=dict(
                    facecolor="palegreen",
                    alpha=0.7,
                    edgecolor="black",
                    boxstyle="round,pad=1",
                ),
            )
            # Add the winning image
            img = IMAGES[winner]
            imagebox = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(
                imagebox,
                (0.42, 0.35),
                frameon=False,
                xycoords="axes fraction",
                box_alignment=(0.5, 0.3),
            )
            ax.add_artist(ab)
            # Add the crown image
            crown_img = mpimg.imread(image_pathroot + "crown.png")
            crown_imagebox = OffsetImage(crown_img, zoom=0.3)
            crown_ab = AnnotationBbox(
                crown_imagebox,
                (0.58, 0.35),
                frameon=False,
                xycoords="axes fraction",
                box_alignment=(0.5, 0.3),
            )
            ax.add_artist(crown_ab)

            ani.event_source.stop()
            # plt.close()

        return artists
    
    # Further hyperparameters
    arena_radius = 10.0
    ball_radius = min(1.0, np.sqrt((4 * arena_radius) * 0.35 / number_balls))
    random_balltype_init = (
        False  # ...when True the ball species are initialised completely randomly
    )

    # Bonus ball hyperparameters
    bonus_ball = False
    bonus_radius = 3 * ball_radius

    # Other hyperparameters
    bounds = [-arena_radius, arena_radius, -arena_radius, arena_radius]
    centre_bounds = [ball_radius - arena_radius, arena_radius - ball_radius]
    dt = 0.1  # ...the timestep of the simulation

    fig, ax = build_plot(bounds)
    balls = build_balls(
        bonus_ball,
        bonus_radius,
        random_balltype_init,
        number_balls,
        max_velocity,
        bounds,
        ball_radius,
    )

    # Create the animation
    max_frames = 200
    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(balls, ax, dt, bounds, centre_bounds),
        frames=max_frames,
        interval=20,
        blit=False,
    )

    # Save the animation to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.gif') as temp_file:
        ani.save(temp_file.name, writer='imagemagick')
        temp_file.seek(0)
        img_data = base64.b64encode(temp_file.read()).decode('utf-8')

    return f'<img src="data:image/gif;base64,{img_data}" />'


if __name__ == "__main__":
    app.run(debug=True)
