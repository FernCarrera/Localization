from tools import make_map,states_to_track
from sensor import NoisySensor
from vehicle import update_state
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
#---------
from filterpy.stats import plot_covariance
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from ukf import *
# ---- debugging
import pdb

fig = plt.figure()
ax = fig.add_subplot(111)
time_text = ax.text(0.02,0.95,'',transform=ax.transAxes)
following_text = ax.text(0.02,0.85,'',transform=ax.transAxes)
position_text = ax.text(0.02,0.75,'',transform=ax.transAxes)
nextp_text = ax.text(0.02,0.65,'',transform=ax.transAxes)
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
nlandmarks = 10

# draw map & get trajectory
path = make_map(map_,landmarks=nlandmarks)  

# Extract starting position
start_x = map_[0][0]
start_y = map_[1][0]
goal = [ map_[0][-1], map_[1][-1] ]

dt = 0.1
u = [0,0]
sim_pos = [0,0,0]
ellipse_step = 20
sigma_range = 0.3; sigma_bearing = 0.1
sigmas = [sigma_range,sigma_bearing]
step = 10

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

sim_pos = list(ukf.x.copy())  # store position of vehicle
sigmas = [sigma_range,sigma_bearing]    # store sensor variances

# set up path dictionary 
x_dict = [round(x,3) for x in path[0]]
y_dict = [round(y,3) for y in path[1]]
path_dict = dict((x,y) for x,y in zip(x_dict,y_dict) )
# get position of landmarks
lmark_pos = make_map.landmarks


# define commands
u = [[0,0]]

counts = 100
   


        
vehicle_pos = []    
p2t = [5,4]
track = [0,0]
def animate(i,dt,u,p2t,track,sim_pos,step,ellipse_step,ukf,sigmas,lmark_pos,goal,path_dict):
    """ Run simulation """
    
    # move vehicle
    sim_pos = move(sim_pos,dt/step,u[-1],wheelbase=0.5)
    
    # do process model prediction
    ukf.predict(u=u[-1],wheelbase=0.5)

    # plot confidence ellipse after process model
    if i % ellipse_step == 0:
        plot_covariance(
                (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                    facecolor='k', alpha=0.3)
        plot_covariance(
                (-10, 10), ukf.P[0:2, 0:2], std=6,
                    facecolor='k', alpha=0.3)

    # store current position    
    state = [round(ukf.x[0],3),round(ukf.x[1],3),round(ukf.x[2],3)    ]
    

    z = []
    z.extend(distance_to(lmark_pos,state,sigmas[0],sigmas[1])) 
    
    # distance and heading of next point to track
    if ([state[0],state[1]] == [start_x,start_y]):
        print('init')
        vtrack,vp2t = point_to_follow(state,path_dict,sigmas[0],sigmas[1])
        track.append(vtrack)
        p2t.append(vp2t)
    if ([state[0],state[1]] == p2t[-1]):
        print('yes')
        vtrack,vp2t = point_to_follow(state,path_dict,sigmas[0],sigmas[1])
        track.append(vtrack)
        p2t.append(vp2t)
        
       
     
    # update heading
    u.append([0.5,track[-1][1]])   
    

    # update the estimated position
    ukf.update(z,landmarks=lmark_pos)

    # plot confidence ellipse after sensor reading and kalman gain
    if i % ellipse_step == 0:
                plot_covariance(
                    (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                    facecolor='g',alpha=0.8)

    vehicle_pos.append([ukf.x[0],ukf.x[1]])
    time_text.set_text('Frame #: %.1f' % i)
    following_text.set_text('Distance to next: {:.2f} Heading: {:.2f}'.format(track[-1][0],track[-1][1]))
    position_text.set_text('Current Position - x: {:.2f} y: {:.2f}'.format(ukf.x[0],ukf.x[1]))
    nextp_text.set_text('Next point - x: {:.2f} y: {:.2f}'.format(p2t[-1][0],p2t[-1][1]))
    

    for t,line in enumerate(lines):
        line.set_data(vehicle_pos[t][0],vehicle_pos[t][1])
        return lines





# run sim
fargs = [dt,u,p2t,track,sim_pos,step,ellipse_step,ukf,sigmas,lmark_pos,goal,path_dict]
ani = FuncAnimation(fig,animate,frames=700,interval=1,repeat=False,fargs= fargs)
plt.show()