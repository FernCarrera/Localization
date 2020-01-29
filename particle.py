from numpy.random import uniform,randn
import numpy as np
import scipy
from study import normalize_angle
import pdb

L = 1.9     # [m] wheel base of vehicle (distance between front and rear axle)
max_steer = np.radians(40.0)    # [rad] max steering agle

def uniform_particles(x_range,y_range,head_range,N):
    """Creates uniformly distributed particles for each
        of the 3 states state with the specified range
        requires modification for more states
    
    Arguments:
        x_range {[float]} -- [variability in x-pos]
        y_range {[float]} -- [variability in y-pos]
        head_range {[float]} -- [variability in heading]
        N {[int]} -- [number of particles]
    
    Returns:
        [numpy array] -- [array of values]
    """

    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(head_range[0], head_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def uniform_particles_single(ranges,N):
    """Test func: diff version of uniform particles
    
    Arguments:
        ranges {[type]} -- [description]
        N {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    particles = np.empty((N,len(ranges)))
    
    for i in np.arange(len(ranges)):
        particles[:,i] = uniform(ranges[i][0],ranges[i][1],size=N)
    # last state must be heading
    particles[:,len(ranges)-1] %= 2 * np.pi 

    return particles


def gaussian_particles(mean,std,N):
    """Generate normally distributed particles
        for 3 states using their mean and std
    
    Arguments:
        mean {[list]} -- [means of every state]
        std {[list]} -- [std of every state]
        N {[int]} -- [number of particles]
    
    Returns:
        [numpy array] -- [array of particles]
    """

    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def predict(particles,u,std,dt=1.):
    """Predict the next position of particles using control
        input with some process noise.
        CONTROL INPUT ORDER MATTERS
    
    Arguments:
        particles {[np.array]} -- [state particles]
        u {[list]} -- [control inputs]
        std {[list]} -- [variance for process noise]
    
    Keyword Arguments:
        dt {[float]} -- [time step] (default: {1.})
    
    Returns:
        [numpy array] -- [particle predictions]
    """

    N = len(particles)
    # update with noise
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # update position w/ process noise
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

    return particles


def predict_2(particles,u,vel,std,dt=1.):
        
    _,delta = u[0],u[1]

    delta = np.clip(delta,-max_steer,max_steer)
    N = len(particles)

    particles[:,0] += vel * np.cos(particles[:,2]) * dt + (randn(N) * std[0])
    particles[:,1] += vel * np.sin(particles[:,2]) * dt + (randn(N) * std[0])
    particles[:,2] += vel / L * np.tan(delta) * dt + (randn(N) * std[1])
    particles[:,2] = normalize_angle(particles[:,2])
    """ 
    self.x += self.v * np.cos(self.yaw) * dt
    self.y += self.v * np.sin(self.yaw) * dt
    self.yaw += self.v / L * np.tan(delta) * dt
    self.yaw = normalize_angle(self.yaw)
    self.v += acceleration * dt
    """
    return particles


def update(particles,weights,z,R,landmarks):
    """Sequential Importance Sampling (SIS)

    Arguments:
        particles {[list]} -- [list of all particles]
        weights {[list]} -- [weight of particles]
        z {[list]} -- [measurements]
        R {[float]} -- [Sensor error]
        landmarks {[[x,y]..]} -- [Position of landmarks]
    """
    #landmarks = np.array(landmarks)
    
    for i, landmarks in enumerate(landmarks):
        distance  = np.linalg.norm(particles[:,0:2] - landmarks,axis=1)
        weights += scipy.stats.norm(distance,R).pdf(z[i])

    weights += 1.e-300  # round off error
    weights /= sum(weights) # normalize
    return weights

def estimate(particles,weights):
    """Compute mean and variance of the particles
    
    Arguments:
        particles {[list]} -- particles
        weights {[list]} -- weight of particles
    
    Returns:
        float,float -- mean and vairance
    """

    pos = particles[:,0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    """Computes the Effective N, which approximately measures the 
       number of particles which meaningfully contribute to the 
       probability distribuition.   

       Use: When this value falls over a specific value its time 
       to resample, this varies by problem but Neff/2 is usually a 
       good place to start.
    
    Arguments:
        weights {[list]} -- [list of all weights]
    """

    return 1. / np.sum(np.square(weights))


def resample_from_index(particles,weights,indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0/len(weights))
    return particles,weights
