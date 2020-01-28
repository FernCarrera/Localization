from tools import make_map,simple_animation
import numpy as np
import matplotlib.pyplot as plt
from study import State,PID,stanley,calc_target_index
from particle import gaussian_particles,predict,update,neff,estimate,resample_from_index
from ukf import distance_to
# -------------- Filter.py ----------------
from filterpy.monte_carlo import stratified_resample



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


    # setup particle filter
    N = 1000
    particles = gaussian_particles([0,0,0],[0.1,0.1,0.1],N)
    weights = np.ones(N) / N    # equal weight to all particles
    position = np.array([state.x,state.y,state.yaw])

   



    xs = []
    NL = 5
    while time <= max_sim_time and last_index > target_index:
        
        # distance & heading to landmark
        #zs = distance_to(lmark_pos,position,0.1,0.1)
        zs = (np.linalg.norm(lmark_pos - [position[0],position[1]]      ,axis=1) + (np.random.randn(NL) * 0.1))
        
        # set up controls
        ai = pd.pid_control(target_speed,state.v,lat_error[-1],time)
        di,target_index = stanley(state,x_path,y_path,target_index)
        state.update(ai,di)

        # predict where particles goin
        predict(particles,(0,0.5),std=(0.1,0.1))
        # combine with measurements
        weights = update(particles,weights,z=zs,R=0.1,landmarks=lmark_pos)

        # check Effective N
        if neff(weights) < N/2:
            indexes = stratified_resample(weights)
            particles,weights = resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        
        time += dt
        
        # store data for plotting
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        lat_error.append(stanley.lat_error)
        position = np.array([state.x,state.y,state.yaw])

        # speed up time if oscilaltions
        if stanley.lat_error > abs(1.5):
            time += 1

        if show_animation:
            Particle = [particles,mu]
            simple_animation(Particle,path,[x,y],lmark_pos,time,max_sim_time)
           

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





