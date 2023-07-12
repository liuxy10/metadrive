import numpy as np


class PointMassModel:
    """
    This model can be used to predict next state
    """
    def __init__(self):
        self.state = dict(x=0, y=0, speed=0, heading_theta=0)

    def reset(self, x, y, speed, heading_theta):
        """
        heading_theta in radian
        """
        self.state = dict(x=x, y=y, speed=speed, heading_theta=heading_theta)

    def step(self, dt, control):
        """
        In this model, we formulate the car's kinematics model as point mass model with heading 
        things need to be finetuned
        mass

        """
        x = self.state["x"]
        y = self.state["y"]
        v = self.state["speed"]
        theta = self.state["heading_theta"]
        # pedal, steering = control[0], control[1]
        acc, heading = control[1], control[0]
        
        new_theta = heading
        new_v = v + acc * dt
        new_x = x + new_v * np.cos(new_theta) * dt
        new_y = y + new_v * np.sin(new_theta) * dt
        new_state = dict(x=new_x, y=new_y, speed=new_v, heading_theta=new_theta)
        self.state = new_state
        return new_state
