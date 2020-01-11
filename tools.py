import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt


def make_map(points,landmarks=0):

    """
        # inputs
        points: [[float,float],...]

        Landmarks: int, # of landmarks

        # returns
        out: [array,array], x,y of track


        Example:
        x = np.arange(0,10,1)   # x coord
        y = random.sample(range(0,20),len(x)) # y coord
        points = np.vstack((x,y))
        make_map(points,5)

    """

    tck,_ = interpolate.splprep(points,s=0) # s=0,gp thru all pts

    unew = np.arange(0,1.01,0.01) # make 101 points for spline to use?
    out = interpolate.splev(unew,tck)
    plt.plot(out[0],out[1],color='orange',label="track")
    #plt.plot(points[:][0],points[:][1],'o',color='blue',label='points')
    
    # make landmarks
    lmarks = make_landmarks(landmarks,out)

    # plot landmarks if theres any
    if len(lmarks) > 0:
        lmarks = np.array(lmarks)
        plt.scatter(lmarks[:,0],lmarks[:,1],marker='P',label='landmarks')
    
    # make variable for storing landmark locations
    make_map.landmarks = lmarks
    
    # draw start/finish line
    plt.scatter(out[0][-1],out[1][-1],marker='x',c='red')
    plt.scatter(out[0][0],out[1][0],marker='2',c='green')
    #plt.plot(lmarks[0],lmarks[1],'P')
    
    plt.legend()
    #plt.show()
    return out

def make_landmarks(num_landmk,mapv):
    """[Generates landmarks based on track]
    
    Arguments:
        num_landmk {[int]} -- [number of landmarks]
        mapv {[list]} -- [points of map]
    
    Returns:
        [[x,y]...] -- [list with points of landmarks]
    """
    lmarks = []
    for _ in range(num_landmk):
        x = random.randint(int(min(mapv[0])),int(max(mapv[0])))
        y = random.randint(int(min(mapv[1])),int(max(mapv[1])))

        lmarks.append([x,y])
    return lmarks

# Animation functions

def states_to_track(nstates,plot_colors,state_names,markers=None,timer=None,fname='fig'):
    """Builds list that is used to animate in matplotlib
    
    Arguments:
        nstates {[int]} -- [Number of states to track]
        plot_colors {[color,color...]} -- [list of color of each state]
        state_names {[str,str...]} -- [names of states for use in legend]
    
    Keyword Arguments:
        markers {[str,str...]} -- [markers for states] (default: {None})
    
    Returns:
        [list] -- [list object used for animation]

        Example:

        nstates = 3 
        plotcols = ["0.8","orange","black"]
        names = ['Sensor Data','Actual Movement','kalman_estimate']
        markers = ['o','_',',']
    """


    if markers == None:
        markers = [None]*nstates

    if timer != None:
        ax = fname.add_subplot(111)
        time_text = ax.text(0.02,0.95,'',transform=ax.transAxes)

    lines = []
    for index in range(nstates):
        state_set = plt.plot([],[],color=plot_colors[index],
                                marker=markers[index],label=state_names[index])[0]
        lines.append(state_set)

    return lines


def init(lines):
    """Sets data to be plotted, called by FuncAnimation in matplotlib
    
    Arguments:
        lines {[list]} -- [Line list returned in states_to_track()]
        time_text {[matplotlib obj]} -- [Update text each epoch]


    text/time example:
        Inside function were the simulation is being simulated
        time_text.set_text('Frame #: %.1f' % frame_number)
    """
    for line in lines:
        line.set_data([],[])

    #time_text.set_text('')

