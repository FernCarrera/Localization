from tools import make_map,states_to_track
from sensor import NoisySensor
from vehicle import update_state
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
#---------
from filterpy.stats import plot_covariance_ellipse
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from ukf import *
# ---- debugging
import pdb

fig = plt.figure()
def plot_track(track):
    ''' draw the path that the vehicle took'''
    track = np.array(track)
    plt.plot(track[:, 0], track[:,1], color='k', lw=2)
    plt.axis('equal')
    plt.title("UKF Robot localization")
    plt.show()

# Create a random points to interpolate into a path
x1 = np.arange(0,10,1)   # x coord
y = random.sample(range(0,20),len(x1)) # y coord
map_ = np.vstack((x1,y))
nlandmarks = 5

# draw map & get trajectory
path = make_map(map_,nlandmarks)  

# Extract starting position
start_x = map_[0][0]
start_y = map_[1][0]
goal = [ map_[0][-1], map_[1][-1] ]

lines = states_to_track(1,['red'],['vehicle'])
def init():

    for line in lines:
        line.set_data([],[])

# compute MSP, and introduce our residual func that normalizes angles
points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0, 
                                subtract=residual_x)

dt = 0.1
# unscented kalman object
#global ukf
ukf = UKF(dim_x=3, dim_z=2*nlandmarks, fx=move, hx=Hx,
            dt=dt, points=points, x_mean_fn=state_mean, 
            z_mean_fn=z_mean, residual_x=residual_x, 
            residual_z=residual_h)

# define UKF matrices
ukf.x = np.array([start_x,start_y,0]) # x,y,heading
ukf.P = np.diag([0.1,0.1,0.5])
ukf.R = np.diag([sigma_range**2, sigma_bearing**2]*nlandmarks)
ukf.Q = np.eye(3)*0.0001



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

    

    '''
    # compute MSP, and introduce our residual func that normalizes angles
    points = MerweScaledSigmaPoints(n=3, alpha=.00001, beta=2, kappa=0, 
                                    subtract=residual_x)

    dt = 0.1
    # unscented kalman object
    ukf = UKF(dim_x=3, dim_z=2*nlandmarks, fx=move, hx=Hx,
              dt=dt, points=points, x_mean_fn=state_mean, 
              z_mean_fn=z_mean, residual_x=residual_x, 
              residual_z=residual_h)
    '''
    

    sim_pos = list(ukf.x.copy())  # store position of vehicle
    sigmas = [sigma_range,sigma_bearing]    # store sensor variances

   
    

    # set up path dictionary 
    x_dict = [round(x,3) for x in path[0]]
    y_dict = [round(y,3) for y in path[1]]
    path_dict = dict((x,y) for x,y in zip(x_dict,y_dict) )
    # get position of landmarks
    lmark_pos = make_map.landmarks
    
   
    # define commands
    u = [0,0]
    counts = 100
    
    # this needs to get shorter.....
    fargs = [dt,u,sim_pos,step,ellipse_step,ukf,sigmas,lmark_pos,goal,path_dict]
    animate(counts,*fargs)
    
    
    return ukf
        
vehicle_pos = []    

def animate(iterations,dt,u,sim_pos,step,ellipse_step,ukf,sigmas,lmark_pos,goal,path_dict):
    """ Run simulation """
    for i in range(iterations):

        # move vehicle
        sim_pos = move(sim_pos,dt/step,u,wheelbase=0.5)
        
        # do process model prediction
        ukf.predict(u=u,wheelbase=0.5)

        # plot confidence ellipse after process model
        if i % ellipse_step == 0:
            plot_covariance_ellipse(
                    (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                     facecolor='k', alpha=0.3)

        # store current position    
        state = ukf.x
        
        if([state[0:2]] == goal):
            print("goal")
            return ukf
        z = []
        z.extend(distance_to(lmark_pos,state,sigmas[0],sigmas[1])) 
        
        # distance and heading of next point to track
        track = point_to_follow(state,path_dict,sigmas[0],sigmas[1])
        
        # update heading
        u = [1,track[1]]    

        # update the estimated position
        ukf.update(z,landmarks=lmark_pos)

        # plot confidence ellipse after sensor reading and kalman gain
        if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                        facecolor='g',alpha=0.8)

        vehicle_pos.append([ukf.x[0],ukf.x[1]])

        for t,line in enumerate(lines):
            line.set_data(vehicle_pos[t][0],vehicle_pos[t][1])
            return lines





dt = 0.1
u = [0,0]
sim_pos = [0,0,0]
ellipse_step = 20
sigma_range = 0.3; sigma_bearing = 0.1
sigmas = [sigma_range,sigma_bearing]

fargs = [dt,u,sim_pos,step,ellipse_step,ukf,sigmas,lmark_pos,goal,path_dict]
ani = FuncAnimation(fig,animate,frames=201,interval=dt,init_func=init,repeat=False)

# ---------
sensor_std = 2.
#track, zs = generate_data(50, sensor_std)



# run localization
'''
localization(
    points, landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
    sigma_range=0.3, sigma_bearing=0.1, step=1,
    ellipse_step=20)
plt.show()
'''
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