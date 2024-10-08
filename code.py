'''
Code to simulate rock-paper-scissors swarm game
...to run:
    ~ ensure filepath for images is correct
    ~ set hyperparameters as desired
    ~ run script
'''
# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties

# Main hyperparameters
number_balls = 10
max_velocity = 1.
save = False
max_frames = 200

# Further hyperparameters
arena_radius = 10.
ball_radius = min(1.,np.sqrt((4*arena_radius)*0.35/number_balls))
random_balltype_init = False #...when True the ball species are initialised completely randomly
bounce = True                #...when False balls only bounce off the boundary and not each other

# Bonus ball hyperparameters
bonus_ball = True
bonus_radius = 3*ball_radius

# Other hyperparameters
bounds = [-arena_radius, arena_radius, -arena_radius, arena_radius]
centre_bounds = [ball_radius-arena_radius, arena_radius-ball_radius]
dt = 0.1 #...the timestep of the simulation

# Define the game rules
rules = {
    'rock': 'scissors',
    'scissors': 'paper',
    'paper': 'rock'
}

# Load images
image_pathroot = './Images/'
images = {
    'rock': mpimg.imread(image_pathroot+'rock.png'),
    'paper': mpimg.imread(image_pathroot+'paper.png'),
    'scissors': mpimg.imread(image_pathroot+'scissors.png'),
    'bonus': mpimg.imread(image_pathroot+'rage.png')
}

# Define font style
font = FontProperties()
font.set_family('copperplate')
font.set_size(30)
font.set_weight('bold')

class Ball:
    def __init__(self, kind, inert=False, radius=1., position=None, velocity=None):
        self.kind = kind
        self.inert = inert
        self.radius = float(radius)
        self.position = np.array(position if position is not None else [0., 0.])
        self.velocity = np.array(velocity if velocity is not None else [0., 0.])
        self.image = images[self.kind]
        self.image_artist = None

    def calculate_extent(self):
       return [
            self.position[0] - self.radius,  # Left
            self.position[0] + self.radius,  # Right
            self.position[1] - self.radius,  # Bottom
            self.position[1] + self.radius   # Top
        ]

    def update_position(self, dt, bounds):
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
        self.image_artist = ax.imshow(self.image, extent=extent, aspect='auto')
        return self.image_artist

def check_collision(ball1, ball2):
    dist = np.linalg.norm(ball1.position - ball2.position)
    return dist < (ball1.radius + ball2.radius)

def resolve_collision(ball1, ball2):
    if not (ball1.inert or ball2.inert):
        if rules[ball1.kind] == ball2.kind:
            ball2.kind = ball1.kind
            ball2.image = images[ball2.kind]
        elif rules[ball2.kind] == ball1.kind:
            ball1.kind = ball2.kind
            ball1.image = images[ball1.kind]
        
    if not bounce:
        return
        
    # Calculate the normal vector
    normal = ball2.position - ball1.position
    normal /= np.linalg.norm(normal)
    
    # Calculate the relative velocity
    relative_velocity = ball2.velocity - ball1.velocity
    
    # Calculate the velocity component along the normal
    velocity_along_normal = np.dot(relative_velocity, normal)
    
    if velocity_along_normal > 0:
        return  # Balls are moving apart, no need to resolve

    # Calculate the new velocities
    ball1.velocity += velocity_along_normal * normal
    ball2.velocity -= velocity_along_normal * normal
    
    # Adjust positions to avoid overlap
    overlap = (ball1.radius + ball2.radius) - np.linalg.norm(ball2.position - ball1.position)
    if overlap > 0: 
        ball1.position -= normal * 1.01 * overlap / 2
        ball2.position += normal * 1.01 * overlap / 2
        
        # Ensure the positions are within the bounding box
        ball1.position = np.clip(ball1.position, *centre_bounds)
        ball2.position = np.clip(ball2.position, *centre_bounds)
        
    return

def animate(i, balls, ax, dt, bounds):
    
    for ball in balls:
        ball.update_position(dt, bounds)
        
    for j in range(len(balls)):
        for k in range(j + 1, len(balls)):
            if check_collision(balls[j], balls[k]):
                resolve_collision(balls[j], balls[k])
    
    artists = []
    for ball in balls:
        artists.append(ball.draw(ax))
        
    # Check if all balls are the same kind
    if bonus_ball:
        kinds = [ball.kind for ball in balls[1:]]
    else:
        kinds = [ball.kind for ball in balls]
    if len(set(kinds)) == 1:
        winner = kinds[-1]
        ax.text(0.5, 0.5, f'{winner.capitalize()} wins!\n\n\n\n', transform=ax.transAxes,
                ha='center', va='center', color='black', fontproperties=font,
                bbox=dict(facecolor='palegreen', alpha=0.7, edgecolor='black', boxstyle='round,pad=1'))
        # Add the winning image
        img = images[winner]
        imagebox = OffsetImage(img, zoom=0.3)
        ab = AnnotationBbox(imagebox, (0.42, 0.45), frameon=False, xycoords='axes fraction', box_alignment=(0.5, 0.4))
        ax.add_artist(ab)
        # Add the crown image
        crown_img = mpimg.imread(image_pathroot+'crown.png')
        crown_imagebox = OffsetImage(crown_img, zoom=0.3)
        crown_ab = AnnotationBbox(crown_imagebox, (0.58, 0.45), frameon=False, xycoords='axes fraction', box_alignment=(0.5, 0.4))
        ax.add_artist(crown_ab)
                
        ani.event_source.stop()
        #plt.close()
    
    return artists

# Example usage
fig, ax = plt.subplots()
fig_manager = plt.get_current_fig_manager()
fig_manager.set_window_title('Rock, Paper, Scissors...')
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[2], bounds[3])
fig.patch.set_facecolor('lavender')  
#ax.axis('off')
ax.xaxis.set_visible(True)  # Ensure the x-axis line is visible
ax.yaxis.set_visible(True)  # Ensure the y-axis line is visible
ax.tick_params(axis='both', which='both', length=0)  # Turn off the ticks
ax.set_xticklabels([])  # Remove x-axis labels
ax.set_yticklabels([])  # Remove y-axis labels


# Define positon overlap check function
def is_overlapping(new_ball, balls):
    for ball in balls:
        distance = np.linalg.norm(new_ball.position - ball.position)
        if distance < (2.01 * ball_radius):
            return True
    return False

# Initialize balls
balls = []
if bonus_ball:
    #balls.append(Ball(kind=list(images.keys())[0], radius=bonus_radius, position=np.zeros(2), velocity=np.zeros(2)))
    balls.append(Ball(kind='bonus', inert=True, radius=bonus_radius, position=np.zeros(2), velocity=np.zeros(2)))

# Random types
if random_balltype_init:
    for ball in range(number_balls):
        overlap = True
        kind = np.random.choice(list(rules.keys()))
        velocity = np.array([np.random.uniform(-max_velocity, max_velocity), np.random.uniform(-max_velocity, max_velocity)])
        while overlap:
            position = np.array([np.random.uniform(bounds[0]+ball_radius*1.01, bounds[1]-ball_radius*1.01), np.random.uniform(bounds[2]+ball_radius*1.01, bounds[3]-ball_radius*1.01)])
            new_ball = Ball(kind=kind, radius=ball_radius, position=position, velocity=velocity)
            overlap = is_overlapping(new_ball, balls)
        balls.append(new_ball)
# Equal types
else:
    for _ in range(number_balls//3):
        for kind in rules.keys():
            overlap = True
            velocity = np.array([np.random.uniform(-max_velocity, max_velocity), np.random.uniform(-max_velocity, max_velocity)])
            while overlap:
                position = np.array([np.random.uniform(bounds[0]+ball_radius*1.01, bounds[1]-ball_radius*1.01), np.random.uniform(bounds[2]+ball_radius*1.01, bounds[3]-ball_radius*1.01)])
                new_ball = Ball(kind=kind, radius=ball_radius, position=position, velocity=velocity)
                overlap = is_overlapping(new_ball, balls)
            balls.append(new_ball)
        
    if number_balls%3 == 1:
        overlap = True
        kind = np.random.choice(list(rules.keys()))
        velocity = np.array([np.random.uniform(-max_velocity, max_velocity), np.random.uniform(-max_velocity, max_velocity)])
        while overlap:
            position = np.array([np.random.uniform(bounds[0]+ball_radius*1.01, bounds[1]-ball_radius*1.01), np.random.uniform(bounds[2]+ball_radius*1.01, bounds[3]-ball_radius*1.01)])
            new_ball = Ball(kind=kind, radius=ball_radius, position=position, velocity=velocity)
            overlap = is_overlapping(new_ball, balls)
        balls.append(new_ball)
        
    elif number_balls%3 == 2:
        kinds = np.random.choice(list(rules.keys()),2,replace=False)
        for kind in kinds:
            overlap = True
            velocity = np.array([np.random.uniform(-max_velocity, max_velocity), np.random.uniform(-max_velocity, max_velocity)])
            while overlap:
                position = np.array([np.random.uniform(bounds[0]+ball_radius*1.01, bounds[1]-ball_radius*1.01), np.random.uniform(bounds[2]+ball_radius*1.01, bounds[3]-ball_radius*1.01)])
                new_ball = Ball(kind=kind, radius=ball_radius, position=position, velocity=velocity)
                overlap = is_overlapping(new_ball, balls)
            balls.append(new_ball)

# Create the animation
ani = animation.FuncAnimation(fig, animate, fargs=(balls, ax, dt, bounds), frames=max_frames, interval=20, blit=False)

# Save the animation
if save:
    seed = np.random.randint(1e4)
    ani.save(f'./runs/RPS_sim_(balls={number_balls},speed={max_velocity})_{seed}.mp4', writer='ffmpeg', fps=30)
        
# Show the animation
plt.show()
