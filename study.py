import numpy as np
import matplotlib.pyplot as plt
import sys
from math import atan2

# add path designer

# default control gains
k = 0.5 # control gain
Kp = 1.0 # proportional gain 
dt = 0.1    # [s] time rate of change
L = 2.9     # [m] wheel base of vehicle
max_steer = np.radians(30.0)    # [rad] max steering agle

show_animation = True


class State(object):
    
    def __init__(self,x=0.0,y=0.0,yaw=0.0,v=0.0):
        """Initialize the state of the vehicle
        
        Keyword Arguments:
            x {float} -- [X-coordinate] (default: {0.0})
            y {float} -- [Y-Coordinate] (default: {0.0})
            yaw {float} -- [Yaw/heading angle] (default: {0.0})
            v {float} -- [velocity] (default: {0.0})
        """
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self,acceleration,delta):
        """Update state of vehicle, using bicycle model

        
        Arguments:
            acceleration {[float]} -- [acceleration]
            delta {[float]} -- [steering angle]
        """
        delta = np.clip(delta,-max_steer,max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt

def pid(target,current):

    return Kp * (target - current)

def stanley_control(state,cx,cy,cyaw,last_target_idx):
    """Implementation of stanley control
    
    Arguments:
        state {[State object]} -- [state of object]
        cx {[float]} -- [description]
        cy {[float]} -- [description]
        cyaw {[float]} -- [description]
        last_target_idx {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    current_target_idx, error_front_axle = calc_target_index(
                    state,cx,cy)
    
    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # used to correct heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # used to correct cross track error
    theta_d = np.arctan2(k * error_front_axle,state.v)
    # steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def stanley_2(state,cx,cy,last_target_idx):
    """Implementation of stanley control
    
    Arguments:
        state {[State object]} -- [state of object]
        cx {[float]} -- [description]
        cy {[float]} -- [description]
        cyaw {[float]} -- [description]
        last_target_idx {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    trgt_idx, error_front_axle = calc_target_index(
                    state,cx,cy)
    
    if last_target_idx >= trgt_idx:
        trgt_idx = last_target_idx

    #print("current_target:{}".format(trgt_idx))
    #print("state_y: {}".format(state.y))
    angle_error = atan2(cy[trgt_idx]-state.y,cx[trgt_idx]-state.x) 

    # used to correct heading error
    theta_e = normalize_angle(angle_error - state.yaw)
    # used to correct cross track error
    theta_d = np.arctan2(k * error_front_axle,state.v)
    # steering control
    delta = theta_e + theta_d

    return delta, trgt_idx


def angle_between(x,y):
    # key = cmp function
    return min(y-x, y-x+360, y-x-360, key=abs)  



def normalize_angle(angle):

    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def calc_target_index(state,cx,cy):

    # calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # search for nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx,dy)
    target_idx = np.argmin(d)

    # project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                        -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx],dy[target_idx]],front_axle_vec)

    return target_idx, error_front_axle

