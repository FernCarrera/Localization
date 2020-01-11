import numpy as np
from math import atan2


#--------------- Math -------------------


def point_to_follow(state,path,sigma_range,sigma_bearing):
    """Uses current location to find the next point in path
    
    Arguments:
        state {[float]} -- [x-position]
        path {[dictionary]} -- [stored path]
    
    Returns:
        [float] -- [y-point in path to follow]
    """
 
    if state[0] in path:
        track =  path[state[0]]
    else:
        track = path[min(path.keys(),key=lambda k: abs(k-state[0]))]

    point_to_track = [state[0],track]
    dist_head = distance_to(point_to_track,state,sigma_range,sigma_bearing)

    """
    print("current location {}".format( [state[0],state[1]]) )
    print("position to track {}".format(point_to_track))
    print("distance & heading to next point {}".format(dist_head))
    """

    return dist_head


    

def angle_between(x,y):
    # key = cmp function
    return min(y-x, y-x+360, y-x-360, key=abs)  

def move(x, dt, u, wheelbase):
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt

        if abs(steering_angle) > 0.001: # is robot turning?
            beta = (dist / wheelbase) * np.tan(steering_angle)
            r = wheelbase / np.tan(steering_angle) # radius

            sinh, sinhb = np.sin(hdg), np.sin(hdg + beta)
            cosh, coshb = np.cos(hdg), np.cos(hdg + beta)
            return x + np.array([-r*sinh + r*sinhb, 
                                r*cosh - r*coshb, beta])
        else: # moving in straight line
            return x + np.array([dist*np.cos(hdg), dist*np.sin(hdg), 0])
        

def turn(v, t0, t1, steps,v2=0):

    if v2==0:
        return [[v, a] for a in np.linspace(np.radians(t0), np.radians(t1), steps)]
    
    else:

        v = np.linspace(v, v2, steps)
        a = np.linspace(np.radians(t0), np.radians(t1), steps) 
        turn_accel = list(zip(v,a))
        turn_accel = [list(x) for x in turn_accel]
        
        return turn_accel


def normalize_angle(x):
    x = x % (2*np.pi) # put in range [0,2pi]
    if x > np.pi:       # move to [-pi,pi]
        x -= 2*np.pi
    return x

def residual_h(a,b):
    """ residual with landmarks"""
    y = a - b
    # data in format [dist_1,bearing_1...dist_n,bearing_n]
    for i in range(0,len(y),2):
        y[i+1] = normalize_angle(y[i+1])
    return y

def residual_x(a, b):
    #print(a)
    #print("----")
    #print(b)
    y = a - b
    y[2] = normalize_angle(y[2])
    return y

def Hx(x,landmarks):
    """ takes a state variable and returns teh measurement
    that would correstpond to that state"""
    hx = []
    for lmark in landmarks:
        px,py = lmark
        dist = np.sqrt((px - x[0])**2 + (py - x[1])**2)
        angle = atan2(py - x[1], px - x[0])
        # normalize because measurement model can give results
        # outside [-pi,pi]
        hx.extend([dist, normalize_angle(angle - x[2])])
    return np.array(hx)


def state_mean(sigmas,Wm):
    """ find averages of the state but with atan2
        aka: arctan of the sum of sines and cosines
    """
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)

    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+1]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+1]), Wm))

        x[z] = np.sum(np.dot(sigmas[:,z], Wm))
        x[z+1] = atan2(sum_sin, sum_cos)
    return x


def distance_to(landmarks,position,sigma_range,sigma_bearing):
    
    # vehicle position
    x,y,head = position[0],position[1],position[2]
    
    if len(landmarks) == 2:
        landmarks = [landmarks]
    

    ret = []
    for lmark in landmarks:
        
        dx, dy = lmark[0] - x, lmark[1] - y
        d = np.sqrt(dx**2 + dy**2) + np.random.randn()*sigma_range
        bearing = atan2(lmark[1] - y, lmark[0] - x)
        a = (normalize_angle(bearing - head + np.random.randn()*sigma_bearing))
        ret.extend([d,a])
      
    
    return ret