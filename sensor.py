from numpy.random import randn

class NoisySensor(object):
    def __init__(self,std_noise=1.0):
        self.std = std_noise

    def sense(self,pos):
        """Pass in actual position as tuple (x, y).       
        Returns position with noise added (x,y)"""
        
        return (pos[0] + (randn() * self.std), 
                pos[1] + (randn() * self.std))