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




    
    



