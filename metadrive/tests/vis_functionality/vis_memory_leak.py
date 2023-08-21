"""
Note: please install memory profiler with: pip install memory_profiler

Usage:

cd this repo
mprof run python vis_memory_leak.py
mprof plot *.dat

in *.dat:
1st column: "MEM" is just a label.

2nd column: RSS (resident set size) in megabytes. 

3rd column: Unix time stamp:


"""

from memory_profiler import profile

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv

from metadrive.policy.replay_policy import  PMKinematicsEgoPolicy

WAYMO_SAMPLING_FREQ = 1/10

class TestMemoryLeakEnv(MetaDriveEnv):
    def __init__(self):
        super(TestMemoryLeakEnv, self).__init__({"manual_control": False, "traffic_density": 0.0, "use_render": False})

    @profile(precision=4, stream=open('memory_leak_test.log', 'w+'))
    def step(self, action):
        return super(TestMemoryLeakEnv, self).step(action)

class TestMemoryLeakEnvWaymo(WaymoEnv):
    def __init__(self, waymo_env_config):
        """Initialize the class.
        
        Args: 
            wrapped_env: the env to be wrapped
            lamb: new_reward = reward + lamb * cost_hazards
        """
        super().__init__(waymo_env_config)

    @profile(precision=4, stream=open('memory_leak_test.log', 'w+'))
    def step(self, action):
        return super().step(action)

def test_waymoEnv(total_steps):
    env = TestMemoryLeakEnvWaymo(
        {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy":PMKinematicsEgoPolicy,
        "waymo_data_directory":'/home/xinyi/src/data/metadrive/pkl_9/',
        "case_num": 10000,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "horizon": 90/5,
        "reactive_traffic": False,
                 "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50)) # 160,
    }
    )
    env.reset()
    
    for i in range(1, total_steps):
        o, r, d, info = env.step([0, 0])
        # env.render("Test: {}".format(i))
        if d:
            env.reset()
        if i%1000 == 0:
            print("at step ", i)
    env.close()

def test_metadriveEnv(total_steps):
    env = TestMemoryLeakEnv()
    env.reset()
    for i in range(1, total_steps):
        o, r, d, info = env.step([0, 1])
        # env.render("Test: {}".format(i))
        if d:
            env.reset()
    env.close()




if __name__ == "__main__":

    # test_waymoEnv(1000000)
    test_metadriveEnv(100000)
    