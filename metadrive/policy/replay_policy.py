import numpy as np
from metadrive.utils.math_utils import wrap_to_pi

from metadrive.policy.base_policy import BasePolicy
has_rendered = False

# class ReplayPolicy(BasePolicy):
#     def __init__(self, control_object, locate_info):
#         super(ReplayPolicy, self).__init__(control_object=control_object)
#         self.traj_info = locate_info["traj"]
#         self.start_index = min(self.traj_info.keys())
#         self.init_pos = locate_info["init_pos"]
#         self.heading = locate_info["heading"]
#         self.timestep = 0
#         self.damp = 0
#         # how many times the replay data is slowed down
#         self.damp_interval = 1
#
#     def act(self, *args, **kwargs):
#         self.damp += self.damp_interval
#         if self.damp == self.damp_interval:
#             self.timestep += 1
#             self.damp = 0
#         else:
#             return [0, 0]
#
#         if str(self.timestep) == self.start_index:
#             self.control_object.set_position(self.init_pos)
#         elif str(self.timestep) in self.traj_info.keys():
#             self.control_object.set_position(self.traj_info[str(self.timestep)])
#
#         if self.heading is None or str(self.timestep - 1) not in self.heading.keys():
#             pass
#         else:
#             this_heading = self.heading[str(self.timestep - 1)]
#             self.control_object.set_heading_theta(np.arctan2(this_heading[0], this_heading[1]) - np.pi / 2)
#
#         return [0, 0]


class ReplayEgoCarPolicy(BasePolicy):
    def __init__(self, control_object, random_seed):
        super(ReplayEgoCarPolicy, self).__init__(control_object=control_object)
        self.trajectory_data = self.engine.traffic_manager.current_traffic_data
        self.traj_info = [
            self.engine.traffic_manager.parse_vehicle_state(
                self.trajectory_data[self.engine.traffic_manager.sdc_index]["state"], i
            ) for i in range(len(self.trajectory_data[self.engine.traffic_manager.sdc_index]["state"]))
        ]
        self.start_index = 0
        self.init_pos = self.traj_info[0]["position"]
        self.heading = self.traj_info[0]["heading"]
        self.timestep = 0
        self.damp = 0
        # how many times the replay data is slowed down
        self.damp_interval = 1

    def act(self, *args, **kwargs):
        self.damp += self.damp_interval
        if self.damp == self.damp_interval:
            self.timestep += 1
            self.damp = 0
        else:
            return [0, 0]

        if self.timestep == self.start_index:
            self.control_object.set_position(self.init_pos)
        elif self.timestep < len(self.traj_info):
            self.control_object.set_position(self.traj_info[int(self.timestep)]["position"])

        if self.heading is None or self.timestep >= len(self.traj_info):
            pass
        else:
            this_heading = self.traj_info[int(self.timestep)]["heading"]
            self.control_object.set_heading_theta(this_heading, rad_to_degree=False)

        return [0, 0]


class PMKinematicsEgoPolicy(BasePolicy):
    def __init__(self, control_object, random_seed):
        super(PMKinematicsEgoPolicy, self).__init__(control_object=control_object)
        # TODO(lijinning): self.dt is set only for waymo dataset.
        # Should be changed to a parameter instead.
        self.dt = 0.1 # waymo dataset timestep

    def act(self,  agent_id):
        action = self.engine.external_actions[agent_id]
        if not self.discrete_action:
            control = action
        else:
            control = self.convert_to_continuous_action(action)

        pos = self.control_object.position
        vel = self.control_object.velocity
        speed = np.linalg.norm(vel)
        heading_theta = self.control_object.heading_theta

        # overwrite dynamics
        state = {
                'x': pos[0],
                'y': pos[1],
                'speed': speed,
                'heading_theta': heading_theta,
        }
        next_state = self.step_point_mass_kinematics(state, control, self.dt)

        self.control_object.set_position([next_state['x'], next_state['y']])
        self.control_object.set_velocity(
            [
                next_state['speed'] * np.cos(next_state['heading_theta']),
                next_state['speed'] * np.sin(next_state['heading_theta']),
            ],
        )
        self.control_object.set_heading_theta(next_state['heading_theta'], rad_to_degree=False)

        return [0, 0]

    @staticmethod
    def step_point_mass_kinematics(
            state: dict[str, float], action: list, dt: float
    ) -> dict[str, float]:
        """Obtain next state by the kinematics model.
        
        Args: 
            state: {pos_x, pos_y, speed, heading_angle}
            action: [heading_angle, acc]
            dt: sampling time
        Returns:
            next_state: the next state.        
        """
        x, y = state["x"], state["y"]
        v = state["speed"]
        # TODO(lijinning): Should we use heading as BC output?
        theta = state["heading_theta"]
        heading, acc = action
        
        new_theta = heading
        new_v = v + acc * dt
        new_x = x + new_v * np.cos(new_theta) * dt
        new_y = y + new_v * np.sin(new_theta) * dt
        new_state = dict(x=new_x, y=new_y, speed=new_v, heading_theta=new_theta)
        return new_state

        