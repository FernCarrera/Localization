import numpy as np
import matplotlib.pyplot as plt
import sys
from math import atan2


# default control gains
k = 0.5 # control gain
Kp = 0.5 # proportional gain 
dt = 0.1    # [s] time rate of change
L = 1.9     # [m] wheel base of vehicle (distance between front and rear axle)
max_steer = np.radians(40.0)    # [rad] max steering agle

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


class PID(object):

    def __init__(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.prev_lat_error = 0.1
        self.prev_state = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.01


    def pid_control(self,target,current,lat_error,time):
        """Proportional Derivative Speed controller
        
        Arguments:
            target {[float]} -- [target speed]
            current {[float]} -- [current speed]
            lat_error {[float]} -- [lateral error]
            time {[float]} -- [time in seconds]
        
        Returns:
            [float] -- [acceleration correction]
        """ 

        if abs(lat_error) > self.prev_lat_error:
            target = 0.5*target # pump the breaks
        
        self.prev_lat_error = abs(lat_error)


        time_elap = time - self.prev_time
        error = target - current
        error_rate = (error - self.prev_error)

        self.prev_error = error
        self.prev_time = time

        d_gain = self.Kd * error_rate/time_elap
        i_gain = self.Ki * (self.prev_state - (error * time_elap))
        p_gain = self.Kp * (target - current)

        return d_gain + p_gain + i_gain


def pid(target,current,lat_error):
    """Proportional speed controller
    
    Arguments:
        target {[float]} -- [target speed]
        current {[float]} -- [current speed]
    
    Returns:
        [float] -- [acceleration needed to correct]
    """
    
    if abs(lat_error) >= 0.5:
        target = 0.4*target
        #prev_error = lat_error
    

    return Kp * (target - current)



def stanley(state,cx,cy,last_target_idx):
    """Implementation of Kinematic Stanley control
    
    Arguments:
        state {[State object]} -- [state of object]
        cx {[float]} -- [x-points of path]
        cy {[float]} -- [y-points of path]
        
        last_target_idx {[type]} -- [description]
    
    Returns:
        float,int -- heading correction and next point to follow
    """

    trgt_idx, error_front_axle = calc_target_index(
                    state,cx,cy)
    
    stanley.lat_error = error_front_axle
    
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

    if type(angle) is np.ndarray:
        for x in angle:
            while x > np.pi:
                
                x -= 2.0 * np.pi

            while x < -np.pi:
                x += 2.0 * np.pi
    else:

        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

    return angle

def calc_target_index(state,cx,cy):
    """Calculate the next point in path to follow
    
    Arguments:
        state {[State object]} -- [State of the vehicle]
        cx {[list]} -- [x-corods of path]
        cy {[list]} -- y-coords of path
    
    Returns:
        int,float -- [index to next target, error axle->target]
    """

    # calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Make triangles between your position and the path 
    # shortest hypothenuse is point to travel to
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx,dy)
    target_idx = np.argmin(d)

    # Calculate error between front axle vector and
    # the target point using dot product
    # neg: pointing in opposite dir
    # 0: vectors are perpendicuar
    # pos: pointing in similar dir 
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                        -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx],dy[target_idx]],front_axle_vec)

    return target_idx, error_front_axle

