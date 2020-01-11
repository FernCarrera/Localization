from tools import make_map
from sensor import NoisySensor
from vehicle import update_state
import numpy as np
import matplotlib.pyplot as plt
import random
#---------
from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from ukf import *
# ---- debugging
import pdb

def plot_track(track):
    ''' draw the path that the vehicle took'''
    track = np.array(track)
    plt.plot(track[:, 0], track[:,1], color='k', lw=2)
    plt.axis('equal')
    plt.title("UKF Robot localization")
    plt.show()

"""
def generate_data(steady_count,std):
    veh = Vehicle(x0=0, y0=0, v0=0.3, heading=0)
    xs, ys = [], []

    for _ in range(30):
        x,y = veh.update()
        xs.append(x)
        ys.append(y)

    veh.set_commanded_heading(310,25)
    veh.set_commanded_speed(1,15)

    for _ in range(steady_count):
        x,y = veh.update()
        xs.append(x)
        ys.append(y)

    ns = NoisySensor(std)
    pos = np.array(list(zip(xs,ys)))
    zs = np.array([ns.sense(p) for p in pos])

    return pos,zs
"""

def localization(map_,nlandmarks,sigma_vel,sigma_steer,
                    sigma_range,sigma_bearing,ellipse_step=1,step=10):
    """[Locates the vehicle using known landmark locations]
    
    Arguments:
        landmarks {[x,y],...} -- [List of landmark x,y points]
        sigma_vel {[float]} -- [Variance in velocity]
        sigma_steer {[float]} -- [variance in steer]
        sigma_range {[float]} -- [variance in range]
        sigma_bearing {[float]} -- [variance in bearing]
    
    Keyword Arguments:
        ellipse_step {int} -- [time step to draw cnf ellipse] (default: {1})
        step {int} -- [How often the UKF runs] (default: {10})
    """

    # Extract starting position
    start_x = map_[0][0]; start_y = map_[1][0]
    goal = [ map_[0][-1], map_[1][-1] ]

    # compute MSP, and introduce our residual func that normalizes angles
    points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0, 
                                    subtract=residual_x)

    dt = 0.1
    # unscneted kalman object
    ukf = UKF(dim_x=3, dim_z=2*nlandmarks, fx=move, hx=Hx,
              dt=dt, points=points, x_mean_fn=state_mean, 
              z_mean_fn=z_mean, residual_x=residual_x, 
              residual_z=residual_h)

    # define UKF matrices
    
    ukf.x = np.array([start_x,start_y,0]) # x,y,heading
    ukf.P = np.diag([0.1,0.1,0.5])
    ukf.R = np.diag([sigma_range**2, sigma_bearing**2]*nlandmarks)
    ukf.Q = np.eye(3)*0.0001

    sim_pos = list(ukf.x.copy())  # store position of vehicle

   
    path = make_map(map_,nlandmarks)  # draw map & get trajectory

    x_dict = [round(x,3) for x in path[0]]
    y_dict = [round(y,3) for y in path[1]]
    path_dict = dict((x,y) for x,y in zip(x_dict,y_dict) )
    
    lmark_pos = make_map.landmarks
    
    i = 0

    # define commands
    u = [0,0]
    counts = 100
    while (i != counts):

        sim_pos = move(sim_pos,dt/step,u,wheelbase=0.5)
        #pdb.set_trace()
        ukf.predict(u=u,wheelbase=0.5)

        if i % ellipse_step == 0:
            plot_covariance_ellipse(
                    (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                     facecolor='k', alpha=0.3)

        # store current position
        #state = sim_pos[0],sim_pos[1],sim_pos[2]      
        state = ukf.x
        
        if([state[0:2]] == goal):
            print("goal")
            return ukf
        z = []
        
        z.extend(distance_to(lmark_pos,state,sigma_range,sigma_bearing)) 
        
        # distance and heading of next point to track
        track = point_to_follow(state,path_dict,sigma_range,sigma_bearing)
        
        # update heading
        u = [1,track[1]]    
        # update the estimated position
        ukf.update(z,landmarks=lmark_pos)
        if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                        facecolor='g',alpha=0.8)
        
        i += 1
        
        
    #plot_track()
    return ukf
        

# ---------
sensor_std = 2.
#track, zs = generate_data(50, sensor_std)

# create the map
x1 = np.arange(0,10,1)   # x coord
y = random.sample(range(0,20),len(x1)) # y coord
points = np.vstack((x1,y))
landmarks = 5

# run localization
localization(
    points, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
    sigma_range=0.3, sigma_bearing=0.1, step=1,
    ellipse_step=20)
plt.show()

"""
plt.figure()
#bp.plot_measurements(*zip(*zs), alpha=0.5)
plt.plot(*zip(*track), color='b', label='track')
plt.scatter(zs[:,0],zs[:,1],alpha=0.5,label='measurements')
plt.axis('equal')
plt.legend(loc=4)
plt.show()
#bp.set_labels(title='Track vs Measurements', x='X', y='Y')
"""