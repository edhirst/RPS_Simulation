import io
from typing import List

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, Response, render_template, request
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from simulation import (IMAGES, Ball, build_balls, build_plot, check_collision,
                        image_pathroot, resolve_collision)

app = Flask(__name__)


def animate(frame, balls: List[Ball], ax, dt, bounds, centre_bounds):
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

    return artists


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_simulation")
def run_simulation():
    try:
        number_balls = int(request.args.get("param1", 10))
        max_velocity = float(request.args.get("param2", 2.0))
        arena_radius = int(request.args.get("param3", 20.0))

        # Error handling for number of balls
        if number_balls < 1:
            number_balls = 1
        elif number_balls > 75:
            number_balls = 75

        # Error handling for max velocity
        if max_velocity < 0.1:
            max_velocity = 0.1
        elif max_velocity > 10.0:
            max_velocity = 10.0

        if arena_radius * 5 < number_balls:
            arena_radius = number_balls / 4
        if arena_radius > 40:
            arena_radius = 40

    except ValueError:
        number_balls = 10
        max_velocity = 2.0
        arena_radius = 20.0

    # Further hyperparameters
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

    def generate_frames():
        while True:
            animate(
                None,
                balls,
                ax,
                dt,
                bounds,
                centre_bounds,
            )  # Update the ball positions
            buf = io.BytesIO()
            plt.savefig(buf, format="jpeg")
            buf.seek(0)
            frame = buf.getvalue()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            buf.close()

    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
