from tools import make_map,states_to_track,plot_covariance_zorder
import numpy as np
import matplotlib.pyplot as plt
from study import *



"""
def plot_track(track):
    ''' draw the path that the vehicle took'''
    track = np.array(track)
    plt.plot(track[:, 0], track[:,1], color='k', lw=2)
    plt.axis('equal')
    plt.title("UKF Robot localization")
    plt.show()"""


def main():

    # Create a random points to interpolate into a path
    x = np.arange(0,30,1)   # x coord
    x = np.append(x,x[::-1])
    y = np.ones(len(x))

    y = [x*2 for x in x[:15]]
    #y.extend([y[-1] for _ in range(0,15)] )
    y.extend( [x*3 for x in x[15:30]])
    y.extend( [x*1.2 for x in x[30:55]])
    y.extend( [x%5 for x in x[55:60]])
    y[-5:] = [-1,-1,-1,-1,-2]
    
    
    map_ = np.vstack((x,y))
    nlandmarks = 10


    # draw map & get trajectory
    path = make_map(map_,landmarks=nlandmarks)
    
    # round path for controller  
    x_path = [round(x,4) for x in path[0][:]]
    y_path = [round(y,4) for y in path[1][:]]
    
    
    state = State(x=x_path[0],y=y_path[0],yaw=np.radians(90.0),v=0.0)
    target_speed = 20.0/3.6     # km/h - > [m/s]

    # Lists to store
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    lat_error = [0.0]

    # calc start tracking value
    target_index,_ = calc_target_index(state,x_path,y_path)
    last_index = len(x_path) - 1
    max_sim_time = 40.0
    dt = 0.1
    time = 0.0
    show_animation = True

    
    while time <= max_sim_time and last_index > target_index:
        ai = pid(target_speed,state.v)
        di,target_index = stanley(state,x_path,y_path,target_index)
        state.update(ai,di)
        #print(last_index,target_index)
        time += dt
        
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        lat_error.append(stanley.lat_error)

        if show_animation:
            plt.cla()   # clear current axes
            # stop simulation with esc key
            plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(path[0][:],path[1][:], color="orange", label = 'course')
            plt.plot(path[0][-1],path[1][-1],marker = 'x', color='red')
            plt.plot(x,y,'-b',label='trajectory')
            plt.text(-40,80,"Time Elapsed:{}".format(round(time,3)))
            plt.text(-40,75,"Time Allotted:{}".format(round(max_sim_time,3)),color='r')

            plt.title("Simulation")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    assert last_index >= target_index, "Cannot reach goal"

    if show_animation:
        plt.plot(path[0][:],path[1][:], ".r", label = 'course')
        plt.plot(x,y,'-b',label='trajectory')
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





