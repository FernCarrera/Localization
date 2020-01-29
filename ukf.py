import numpy as np
from math import atan2


#--------------- Math -------------------


def point_to_follow(state,path,sigma_range,sigma_bearing):
    """Uses current location to find the next point in path
    
    Arguments:
        state {[float,float,float]} -- [vehicle state]
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

    return dist_head,point_to_track


    

def angle_between(x,y):
    # key = cmp function
    return min(y-x, y-x+360, y-x-360, key=abs)  

def move(x, dt, u, wheelbase):
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        #print(vel)
        dist = vel * dt
        

        if abs(steering_angle) > 0.001: # is robot turning?
            beta = (dist / wheelbase) * np.tan(steering_angle)
            r = wheelbase / np.tan(steering_angle) # radius

            sinh, sinhb = np.sin(hdg), np.sin(hdg + beta)
            cosh, coshb = np.cos(hdg), np.cos(hdg + beta)

            loc = x + np.array([-r*sinh + r*sinhb, 
                                r*cosh - r*coshb, beta])
            
            return loc
        else: # moving in straight line
            loc = x + np.array([dist*np.cos(hdg), dist*np.sin(hdg), 0])
            return loc

def f_move(x, dt, u, wheelbase):
        hdg = x[2]
        vel = u[0]
        steering_angle = u[1]
        f_dist = vel*1

        if abs(steering_angle) > 0.001: # is robot turning?
           
            # future prediction
            f_beta = (f_dist / wheelbase) * np.tan(steering_angle)
            f_r = wheelbase / np.tan(steering_angle)

         
            f_sinh, f_sinhb = np.sin(hdg), np.sin(hdg + f_beta)
            f_cosh, f_coshb = np.cos(hdg), np.cos(hdg + f_beta)

            
            f_pred = x + np.array([-f_r*f_sinh + f_r*f_sinhb, 
                                f_r*f_cosh - f_r*f_coshb, f_beta])
            return f_pred
        else: # moving in straight line

            f_pred = x + np.array([f_dist*np.cos(hdg), f_dist*np.sin(hdg), 0])
            return f_pred
        

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

def dist(point,position,sigma_range,sigma_bearing):
     
    # vehicle position
    x,y,head = position[0],position[1],position[2]

    dx, dy = point[0] - x, point[1] - y
    d = np.sqrt(dx**2 + dy**2) + np.random.randn()*sigma_range
    bearing = atan2(point[1] - y, point[0] - x)
    a = (normalize_angle(bearing - head + np.random.randn()*sigma_bearing))
    ret = [d,a]

    return ret


def dist_formula(pos,pt1): 

    return np.sqrt( (pt1[0]-pos[0])**2 + (pt1[1]-pos[1])**2 )


def closest_point(pos,path,bumper):
    # TODO if multiple points with same distance, pick rightmost??
    """Returns the closest point on that path relative
       to current location
    
    Arguments:
        pos {[x,y,heading]} -- [vehicle position]
        path {dictionary} -- [path]
        bumper {[int/float]} -- [how many points around x-location 
                                    do you want to search]
    
    Returns:
        [[x,y]] -- [closest point]
    """
    r_pos = int(pos[0]) + bumper
    l_pos = int(pos[0]) - bumper

    current = 10000
    for x in range(l_pos,r_pos): # values xlocation +- bumper
        
        if x in path:   # if x-pos is in path dictionary
            y =  path[x]
        else:
            # if its not, pick the closest point to x
            y = path[min(path.keys(),key=lambda k: abs(k-x))]

        # find closest point to pos
        point = [x,y]
        dist = dist_formula(point,pos)
        if dist < current:
            closest_point = point
    
    return closest_point
    

    
