import numpy as np
import cv2, math, time, random
import keyboard as k
import torch

class CartPoleEnvironment:
    def __init__(self, gravity=20, mass_cart=1.0, mass_pole=0.1, length=0.5, force_mag=15.0, dt=0.02):
        self.gravity = gravity
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.length = length
        self.force_mag = force_mag
        self.dt = dt

        self.resetEnv()

    def resetEnv(self):
        # Initialize state variables
        self.x = 0.0  # Cart position
        self.x_dot = 0.0  # Cart velocity
        self.theta = np.pi + (random.random()-.5)/100  # Pole angle (upright)
        self.theta_dot = 0.0  # Pole angular velocity
        self.t = 0
        return (self.x, self.theta-np.pi, self.x_dot, self.theta_dot), [1,1,1]

    def nextFrame(self, action, display=False):
        action -= 1
        
        # Apply force to the cart
        force = self.force_mag * action

        # Update cart position and velocity
        x_acc = (force + self.mass_pole * self.length * (self.theta_dot ** 2) * np.sin(self.theta)) / (self.mass_cart + self.mass_pole)
        self.x_dot += x_acc * self.dt
        self.x += self.x_dot * self.dt * 2
        self.x_dot *= .9

        # Update pole angle and angular velocity
        theta_acc = (-self.gravity * np.sin(self.theta) + 1.5 * np.cos(self.theta) * x_acc) / self.length
        self.theta_dot += theta_acc * self.dt
        self.theta += self.theta_dot * self.dt

        self.t += 1

        # Check if the episode is done (pole falls or cart goes out of bounds)
        done = self.x < -2.5 or self.x > 2.5 or self.theta < .5*np.pi or self.theta > 1.5*np.pi

        # Calculate reward (1 for each step)
        reward = 1.0 if not done else -1.0

        if display: self.displayEnv()

        return (self.x, self.theta-np.pi, self.x_dot, self.theta_dot), reward, [1,1,1], done

    def displayEnv(self):
        framerate = 20
        size = 800
        pos = (int((self.x*size/2.5*.3)+size/2), 400)
        l = 100
        start,end = pos, (int(pos[0]-l*math.sin(self.theta)),int(pos[1]+l*math.cos(self.theta)))
        
        img = np.array([[[255]]],dtype=np.uint8)
        img = img.repeat(size,axis=0).repeat(size,axis=1)
        img = cv2.circle(img,(150,400),0,(0),5)
        img = cv2.circle(img,(650,400),0,(0),5)
        img = cv2.line(img,start,end,(0),2)
        cv2.imshow('img',img)
        cv2.waitKey(math.ceil(1000/framerate))
        #cv2.waitKey(1)

    def convState(self, state):
        return torch.tensor([state], dtype=torch.float32)


if __name__ == '__main__':
    # Example usage:
    env = CartPoleEnvironment()

    for _ in range(1000):
        usr = 0
        if k.is_pressed('a') or k.is_pressed('left'): usr = -1
        if k.is_pressed('d') or k.is_pressed('right'): usr = 1
        
        state, reward, _, done = env.nextFrame(usr+1)
        env.displayEnv()
        #print("State:", state, "Reward:", reward, "Done:", done)
        if done:
            print("Episode terminated.")
            break
