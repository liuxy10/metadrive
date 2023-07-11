import logging
from collections.abc import Iterable

from metadrive.component.vehicle_module.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math_utils import clip, wrap_to_pi


class EnvInputPolicy(BasePolicy):
    def __init__(self, obj, seed):
        # Since control object may change
        super(EnvInputPolicy, self).__init__(control_object=None)
        self.discrete_action = self.engine.global_config["discrete_action"]
        self.steering_unit = 2.0 / (
            self.engine.global_config["discrete_steering_dim"] - 1
        )  # for discrete actions space
        self.throttle_unit = 2.0 / (
            self.engine.global_config["discrete_throttle_dim"] - 1
        )  # for discrete actions space
        self.discrete_steering_dim = self.engine.global_config["discrete_steering_dim"]
        self.discrete_throttle_dim = self.engine.global_config["discrete_throttle_dim"]

    def act(self, agent_id):
        action = self.engine.external_actions[agent_id]

        if not self.discrete_action:
            to_process = action
        else:
            to_process = self.convert_to_continuous_action(action)

        # clip to -1, 1
        action = [clip(to_process[i], -1.0, 1.0) for i in range(len(to_process))]

        return action

    def convert_to_continuous_action(self, action):
        if isinstance(action, Iterable):
            assert len(action) == 2
            steering = action[0] * self.steering_unit - 1.0
            throttle = action[1] * self.throttle_unit - 1.0
        else:
            steering = float(action % self.discrete_steering_dim) * self.steering_unit - 1.0
            throttle = float(action // self.discrete_steering_dim) * self.throttle_unit - 1.0

            # print("Steering: ", steering, " Throttle: ", throttle)
        return steering, throttle


class EnvInputHeadingAccPolicy(EnvInputPolicy):

    def __init__(self, obj, seed, disable_clip=True):
        super(EnvInputHeadingAccPolicy, self).__init__(obj, seed)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.obj = obj
        self.disable_clip = disable_clip

    
    def act(self, agent_id):
        """The action space of this policy is [heading angle, acc].

        The input `heading angle` needs to be converted to steering angle, following
        the definition of metadrive physical engine.
        NOTE: the heading angle's unit must be RADIUS!

        Args:
            agent_id: the ID of the ego car.
        
        Returns:
            action: [steering, acc]
        """
        action = self.engine.external_actions[agent_id]

        if not self.discrete_action:
            to_process = action
        else:
            to_process = self.convert_to_continuous_action(action)

        steering = self.steering_control(to_process[0])
        acc = to_process[1]
        action = [steering, acc]
        if self.disable_clip:
            pass
        else:
            action = self.postprocess_action(action) 
        self.action_info["action"] = action
        return action

    def postprocess_action(self, action):
        for i in range(len(action)):
            if -1. < action[i] < 1.:
                logging.warning(
                    f"Action {str(i)} == {str(action[i])} is out of bound!"
                    " Clipped to [-1., 1.]."
                )
                action[i] = clip(action[i], -1., 1.)
        return action


    def steering_control(self, target_heading: float) -> float:
        """Convert target heading to steering angle by a PID.

        Args: 
            target_heading: the target heading angle, which unit is radius.
        """
        ego_vehicle = self.obj
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(target_heading - wrap_to_pi(v_heading))
        return float(steering)
