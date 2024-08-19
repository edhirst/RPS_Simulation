"""
Code to simulate rock-paper-scissors swarm game
...to run:
    ~ ensure filepath for images is correct
    ~ set hyperparameters as desired
    ~ run script
"""

from io import BytesIO
from typing import Dict, List

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# Import libraries
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Define the game rules
RULES: Dict[str, str] = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

# Load images
image_pathroot = "./Images/"
IMAGES = {
    "rock": mpimg.imread(image_pathroot + "rock.png"),
    "paper": mpimg.imread(image_pathroot + "paper.png"),
    "scissors": mpimg.imread(image_pathroot + "scissors.png"),
}


class Ball:
    def __init__(self, kind, radius=1.0, position=None, velocity=None):
        self.kind = kind
        self.radius = float(radius)
        self.position = np.array(position if position is not None else [0.0, 0.0])
        self.velocity = np.array(velocity if velocity is not None else [0.0, 0.0])
        self.image = IMAGES[self.kind]
        self.image_artist = None

    def calculate_extent(self):
        return [
            self.position[0] - self.radius,  # Left
            self.position[0] + self.radius,  # Right
            self.position[1] - self.radius,  # Bottom
            self.position[1] + self.radius,  # Top
        ]

    def update_position(self, dt: int, bounds: List[int]):
        self.position += self.velocity * dt
        extent = self.calculate_extent()

        # Check for collision with bounds and update velocity if necessary
        if extent[0] < bounds[0] or extent[1] > bounds[1]:
            self.velocity[0] = -self.velocity[0]
        if extent[2] < bounds[2] or extent[3] > bounds[3]:
            self.velocity[1] = -self.velocity[1]

    def draw(self, ax):
        if self.image_artist:
            self.image_artist.remove()  # Remove the previous image
        extent = self.calculate_extent()
        self.image_artist = ax.imshow(self.image, extent=extent, aspect="auto")
        return self.image_artist


def check_collision(ball1: Ball, ball2: Ball):
    dist = np.linalg.norm(ball1.position - ball2.position)
    return dist < (ball1.radius + ball2.radius)


def resolve_collision(ball1: Ball, ball2: Ball, centre_bounds: List[float]):
    if RULES[ball1.kind] == ball2.kind:
        ball2.kind = ball1.kind
        ball2.image = IMAGES[ball2.kind]
    elif RULES[ball2.kind] == ball1.kind:
        ball1.kind = ball2.kind
        ball1.image = IMAGES[ball1.kind]

    # Calculate the normal vector
    normal = ball2.position - ball1.position
    normal /= np.linalg.norm(normal)

    relative_velocity = ball2.velocity - ball1.velocity

    # Calculate the velocity component along the normal
    velocity_along_normal = np.dot(relative_velocity, normal)

    if velocity_along_normal > 0:
        return  # Balls are moving apart, no need to resolve

    # Calculate the new velocities
    ball1.velocity += velocity_along_normal * normal
    ball2.velocity -= velocity_along_normal * normal

    # Adjust positions to avoid overlap
    overlap = (ball1.radius + ball2.radius) - np.linalg.norm(
        ball2.position - ball1.position
    )
    if overlap > 0:
        ball1.position -= normal * 1.01 * overlap / 2
        ball2.position += normal * 1.01 * overlap / 2

        # Ensure the positions are within the bounding box
        ball1.position = np.clip(ball1.position, *centre_bounds)
        ball2.position = np.clip(ball2.position, *centre_bounds)

    return


# Define positon overlap check function
def is_overlapping(new_ball: Ball, balls: List[Ball]):
    for ball in balls:
        distance = np.linalg.norm(new_ball.position - ball.position)
        if distance < (2.01 * new_ball.radius):
            return True
    return False


def build_plot(bounds: List[float]):
    # Example usage
    fig, ax = plt.subplots()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title("Rock, Paper, Scissors...")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    fig.patch.set_facecolor("lavender")
    # ax.axis('off')
    ax.xaxis.set_visible(True)  # Ensure the x-axis line is visible
    ax.yaxis.set_visible(True)  # Ensure the y-axis line is visible
    ax.tick_params(axis="both", which="both", length=0)  # Turn off the ticks
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels

    return fig, ax


def build_balls(
    bonus_ball: bool,
    bonus_radius: float,
    random_balltype_init: bool,
    number_balls: int,
    max_velocity: float,
    bounds: List[float],
    ball_radius: float,
):
    # Initialize balls
    balls = []
    if bonus_ball:
        balls.append(
            Ball(
                kind=list(IMAGES.keys())[0],
                radius=bonus_radius,
                position=np.zeros(2),
                velocity=np.zeros(2),
            )
        )

    # Random types
    if random_balltype_init:
        for _ in range(number_balls):
            overlap = True
            kind = np.random.choice(list(IMAGES.keys()))
            velocity = np.array(
                [
                    np.random.uniform(-max_velocity, max_velocity),
                    np.random.uniform(-max_velocity, max_velocity),
                ]
            )
            while overlap:
                position = np.array(
                    [
                        np.random.uniform(
                            bounds[0] + ball_radius * 1.01,
                            bounds[1] - ball_radius * 1.01,
                        ),
                        np.random.uniform(
                            bounds[2] + ball_radius * 1.01,
                            bounds[3] - ball_radius * 1.01,
                        ),
                    ]
                )
                new_ball = Ball(
                    kind=kind, radius=ball_radius, position=position, velocity=velocity
                )
                overlap = is_overlapping(new_ball, balls)
            balls.append(new_ball)
    # Equal types
    else:
        for _ in range(number_balls // 3):
            for kind in RULES.keys():
                overlap = True
                velocity = np.array(
                    [
                        np.random.uniform(-max_velocity, max_velocity),
                        np.random.uniform(-max_velocity, max_velocity),
                    ]
                )
                while overlap:
                    position = np.array(
                        [
                            np.random.uniform(
                                bounds[0] + ball_radius * 1.01,
                                bounds[1] - ball_radius * 1.01,
                            ),
                            np.random.uniform(
                                bounds[2] + ball_radius * 1.01,
                                bounds[3] - ball_radius * 1.01,
                            ),
                        ]
                    )
                    new_ball = Ball(
                        kind=kind,
                        radius=ball_radius,
                        position=position,
                        velocity=velocity,
                    )
                    overlap = is_overlapping(new_ball, balls)
                balls.append(new_ball)

        if number_balls % 3 == 1:
            overlap = True
            kind = np.random.choice(list(IMAGES.keys()))
            velocity = np.array(
                [
                    np.random.uniform(-max_velocity, max_velocity),
                    np.random.uniform(-max_velocity, max_velocity),
                ]
            )
            while overlap:
                position = np.array(
                    [
                        np.random.uniform(
                            bounds[0] + ball_radius * 1.01,
                            bounds[1] - ball_radius * 1.01,
                        ),
                        np.random.uniform(
                            bounds[2] + ball_radius * 1.01,
                            bounds[3] - ball_radius * 1.01,
                        ),
                    ]
                )
                new_ball = Ball(
                    kind=kind, radius=ball_radius, position=position, velocity=velocity
                )
                overlap = is_overlapping(new_ball, balls)
            balls.append(new_ball)

        elif number_balls % 3 == 2:
            kinds = np.random.choice(list(IMAGES.keys()), 2, replace=False)
            for kind in kinds:
                overlap = True
                velocity = np.array(
                    [
                        np.random.uniform(-max_velocity, max_velocity),
                        np.random.uniform(-max_velocity, max_velocity),
                    ]
                )
                while overlap:
                    position = np.array(
                        [
                            np.random.uniform(
                                bounds[0] + ball_radius * 1.01,
                                bounds[1] - ball_radius * 1.01,
                            ),
                            np.random.uniform(
                                bounds[2] + ball_radius * 1.01,
                                bounds[3] - ball_radius * 1.01,
                            ),
                        ]
                    )
                    new_ball = Ball(
                        kind=kind,
                        radius=ball_radius,
                        position=position,
                        velocity=velocity,
                    )
                    overlap = is_overlapping(new_ball, balls)
                balls.append(new_ball)

    return balls
