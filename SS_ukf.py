from tools import make_map,simple_animation
import numpy as np
import matplotlib.pyplot as plt
from study import State,PID,stanley,calc_target_index
from ukf import distance_to,residual_x,move, Hx,z_mean,residual_h,state_mean
from study import L as wheelbase
# -------------- Filter.py ----------------
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.stats import plot_covariance_ellipse


def main():

    # Create a random points to interpolate into a path
    x = np.arange(0,30,1)   # x coord
    x = np.append(x,x[::-1])
    y = np.ones(len(x))
    # designing course
    # TODO: function to create courses
    y = [x*2 for x in x[:15]]
    y.extend( [x*3 for x in x[15:30]])
    y.extend( [x*1.2 for x in x[30:55]])
    y.extend( [x%5 for x in x[55:60]])
    y[-5:] = [-1,-1,-1,-1,-2]
    


    # prepare map for func
    map_ = np.vstack((x,y))
    
    # draw map,landmarks & get trajectory
    path = make_map(map_,landmarks=5,random_seed=42)
    lmark_pos = np.array(make_map.landmarks)
   
    # round path for controller  
    x_path = [round(x,4) for x in path[0][:]]
    y_path = [round(y,4) for y in path[1][:]]
    
    # state and PID object
    state = State(x=x_path[0],y=y_path[0],yaw=np.radians(90.0),v=0.0)
    pd = PID(Kp=0.5,Ki=0.1,Kd=-0.15)
    
    target_speed = 30.0/3.6     # km/h - > [m/s]

    # Lists to store
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    lat_error = [0.0]

    # Setup Initial values
    target_index,_ = calc_target_index(state,x_path,y_path)
    last_index = len(x_path) - 1
    max_sim_time = 50.0
    dt = 0.1
    time = 0.0
    show_animation = True

    # setup UKF
    points = MerweScaledSigmaPoints(n=3,alpha=1e-4,kappa=0.0,beta=2,subtract=residual_x)

    sigma_range = 0.3
    sigma_bearing = 0.1
    

    ukf =   UKF(dim_x=3, dim_z=2*len(lmark_pos), fx=move, hx=Hx,
              dt=dt, points=points, x_mean_fn=state_mean, 
              z_mean_fn=z_mean, residual_x=residual_x, 
              residual_z=residual_h)

    ukf.x = np.array([2, 6, .3])
    ukf.P = np.diag([.1, .1, .05])
    ukf.R = np.diag([sigma_range**2,sigma_bearing**2]*5) # 5 landmarks

    ukf.Q = np.eye(3)*0.0001



    while time <= max_sim_time and last_index > target_index:
        #TODO: make stanley model work with ukf
        ukf.predict(u=u,wheelbase=wheelbase)
        if time % 10 == 0:
            plot_covariance_ellipse(
                    (ukf.x[0], ukf.x[1]), ukf.P[0:2, 0:2], std=6,
                     facecolor='k', alpha=0.3)
        
        # set up controls
        ai = pd.pid_control(target_speed,state.v,lat_error[-1],time)
        di,target_index = stanley(state,x_path,y_path,target_index)
        state.update(ai,di)

        
        
        time += dt
        
        # store data for plotting
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        lat_error.append(stanley.lat_error)
        

        # speed up time if oscilaltions
        if stanley.lat_error > abs(1.5):
            time += 1

        if show_animation:
            simple_animation(path,[x,y],lmark_pos,time,max_sim_time)
           

    assert last_index >= target_index, "Cannot reach goal"

    if show_animation:
        #plt.plot(path[0][:],path[1][:], ".r", label = 'course')
        #plt.plot(x,y,'-b',label='trajectory')
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.grid(True)

        _, (ax1,ax2) = plt.subplots(2,sharex='row')
        ax1.plot(t, [iv * 3.6 for iv in v], "-r")
        ax1.set_ylabel("Speed[km/h]")
        ax1.grid(True)
        ax2.plot(t,lat_error,label='lateral error to next [m]')
        ax2.set_ylabel("Lateral Error")
        ax2.set_xlabel("Time[s]")
        plt.grid(True)
        plt.show()
        

if __name__ == "__main__":
    main()





