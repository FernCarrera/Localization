import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt
from math import cos,sin
from filterpy.stats import covariance_ellipse
from matplotlib.pyplot import figure


def make_map(points,radius=10,landmarks=0,plot_landmarks=True,random_seed=None):
    """[Draws the path]
    
    Arguments:
        points {[float/int]} -- [points to interpolate path on]
    
    Keyword Arguments:
        radius {int} -- [radius of path] (default: {10})
        landmarks {int} -- [number of landmarks to draw] (default: {0})
    
    Returns:
        [ float list] -- [path points]

    Example:
        x = np.arange(0,10,1)   # x coord
        y = random.sample(range(0,20),len(x)) # y coord
        points = np.vstack((x,y))
        make_map(points,5)
        plt.show()
    """
    if random_seed is not None and type(random_seed) is int:
        random.seed(random_seed)

    tck,_ = interpolate.splprep(points,s=0) # s=0,gp thru all pts

    unew = np.arange(0,1.01,0.01) # make 101 points for spline to use?
    out = interpolate.splev(unew,tck)
    #plt.plot(out[0],out[1],color='orange',lw=radius,zorder=1)
    #plt.plot(out[0],out[1],color='red',label="track",zorder=1)
    #plt.plot(points[:][0],points[:][1],'o',color='blue',label='points')

    
    # make landmarks
    lmarks = make_landmarks(landmarks,out)
    make_map.landmarks = lmarks

    # plot landmarks if theres any
    
    if len(lmarks) > 0 and plot_landmarks:
        lmarks = np.array(lmarks)
        plt.scatter(lmarks[:,0],lmarks[:,1],marker='P',label='landmarks',zorder=2)
    
    # make variable for storing landmark locations
    make_map.landmarks = lmarks
    
    # draw start/finish line
    #plt.scatter(out[0][-1],out[1][-1],marker='x',c='red',zorder=2,s=100)
    #plt.scatter(out[0][0],out[1][0],marker='2',c='green',zorder=2,s=100)
    #plt.plot(lmarks[0],lmarks[1],'P')
    
    #plt.legend()
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


def simple_animation(path,pos,lmark_pos,time,max_sim_time,pause=0.001,Particle=None,plot_particles=False):
    if Particle is not None:
        particles = Particle[0]
        mu = Particle[1]
    
    plt.cla()   # clear current axes

    # stop simulation with esc key
    plt.gcf().canvas.mpl_connect('key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    
    plt.plot(path[0][:],path[1][:], color="orange", label = 'course')
    plt.plot(path[0][-1],path[1][-1],marker = 'x', color='red')
    plt.plot(pos[0],pos[1],'-b',label='Vehicle')
    plt.scatter(lmark_pos[:,0],lmark_pos[:,1],marker='P',label='landmarks')
    plt.text(-40,80,"Time Elapsed:{}".format(round(time,3)))
    plt.text(-40,75,"Time Allotted:{}".format(round(max_sim_time,3)),color='r')

    if plot_particles:
        plt.scatter(particles[:, 0], particles[:, 1], 
                    color='k', marker=',',label="particles",alpha=0.5, s=1)
        #p1 = plt.scatter(position[0], position[1], marker='+',
        #        color='k', s=180, lw=3)
        plt.scatter(mu[0], mu[1], marker='s',label="estimated pos", color='r')


    plt.title("Simulation")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.pause(pause)




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


def plot_covariance_zorder(
        mean, cov=None, variance=1.0, std=None, interval=None,
        ellipse=None, title=None, axis_equal=True,
        show_semiaxis=False, show_center=True,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid',zorder=1):
 
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt

    if cov is not None and ellipse is not None:
        raise ValueError('You cannot specify both cov and ellipse')

    if cov is None and ellipse is None:
        raise ValueError('Specify one of cov or ellipse')

    if facecolor is None:
        facecolor = fc

    if edgecolor is None:
        edgecolor = ec

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title(title)

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    std = _std_tuple_of(variance, std, interval)
    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2, ls=ls,zorder=zorder)
        ax.add_patch(e)
    x, y = mean
    if show_center:
        plt.scatter(x, y, marker='+', color=edgecolor)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height/4, width/4
        plt.plot([x, x+ h*cos(a+np.pi/2)], [y, y + h*sin(a+np.pi/2)])
        plt.plot([x, x+ w*cos(a)], [y, y + w*sin(a)])


def _std_tuple_of(var=None, std=None, interval=None):
    """
    by: @rlabbe

    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.
    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)
    """

    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std


    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var)


