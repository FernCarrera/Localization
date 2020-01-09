from math import sin, cos, radians
import numpy as np

def angle_between(x,y):
    # key = cmp function
    return min(y-x, y-x+360, y-x-360, key=abs)  


def update_state(x,dt,cmd,steps=30):
    
    """
    x shape: [x,y,vel,heading]
    cmd shape: [vel,heading]
    """

    vx = x[2] * cos(radians(90-x[3]))
    vy = x[2] * sin(radians(90-x[3]))

    # euler integration
    x[0] += vx
    x[1] += vy

    cmd_head = cmd[1]
    cmd_vel = cmd[0]

    # calculate change in heading
    head_delta = angle_between(cmd_head,x[3])
    if abs(head_delta) > 0:
        head_step = steps
    else: 
        head_step = 0

    vel_delta = (cmd_vel - x[2])/steps
    if abs(vel_delta) > 0:
        vel_step = steps
    else:
        vel_step = 0

    # steps is analogous to # of propagations
    if head_step > 0:

        head_step -= 1
        x[3] += head_delta

    if vel_step > 0:
        
        vel_step -= 1
        x[2] += vel_delta

    return x



class Vehicle(object):
    def __init__(self,x0,y0,v0,heading):
        self.x = x0
        self.y = y0
        self.vel = v0
        self.head = heading

        self.cmd_vel = v0
        self.cmd_head = heading
        self.vel_step = 0
        self.head_step = 0
        self.vel_delta = 0
        self.head_delta = 0

    def update(self):
        vx = self.vel * cos(radians(90-self.head))
        vy = self.vel * sin(radians(90-self.head))
        
        # Euler integration
        self.x += vx
        self.y += vy

        # steps is analogous to # of propagations
        if self.head_step > 0:

            self.head_step -= 1
            self.head += self.head_delta

        if self.vel_step > 0:
            
            self.vel_step -= 1
            self.vel += self.vel_delta

        return (self.x,self.y)

    def set_commanded_heading(self,head_deg,steps):
        """ Computes the delta head
            by diving the commanded heading by the 
            amoung of steps you want the propagation
            to occur in.

        """
        
        self.cmd_head = head_deg
        # complete heading change by the number of steps
        self.head_delta = angle_between(self.cmd_head,self.head)/steps

        if abs(self.head_delta) > 0:
            self.head_step = steps
        else:
            self.head_step = 0

    def set_commanded_speed(self, speed, steps):
        """ Computes the delta speed
            by diving the commanded heading by the 
            amoung of steps you want the propagation
            to occur in.
        """
        self.cmd_vel = speed
        self.vel_delta = (self.cmd_vel - self.vel) / steps
        if abs(self.vel_delta) > 0:
            self.vel_step = steps
        else:
            self.vel_step = 0 




