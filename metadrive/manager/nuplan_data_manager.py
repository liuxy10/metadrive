import logging
import os
import tempfile
from os.path import join
import hydra
import nuplan.planning.script.builders.worker_pool_builder
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.utils import set_up_common_builder
from dataclasses import dataclass
from metadrive.manager.base_manager import BaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(__file__)


# Copied from nuplan-devkit tutorial_utils.py
@dataclass
class HydraConfigPaths:
    """
    Stores relative hydra paths to declutter tutorial.
    """

    common_dir: str
    config_name: str
    config_path: str
    experiment_dir: str


def construct_simulation_hydra_paths(base_config_path: str):
    """
    Specifies relative paths to simulation configs to pass to hydra to declutter tutorial.
    :param base_config_path: Base config path.
    :return: Hydra config path.
    """
    common_dir = "file://" + join(base_config_path, 'config', 'common')
    config_name = 'default_simulation'
    config_path = join(base_config_path, 'config', 'simulation')
    experiment_dir = "file://" + join(base_config_path, 'experiments')
    return HydraConfigPaths(common_dir, config_name, config_path, experiment_dir)


class NuPlanDataManager(BaseManager):
    """
    This manager serves as the interface between nuplan-devkit and MetaDrive
    """
    NUPLAN_PACKAGE_PATH = os.path.dirname(nuplan.__file__)
    NUPLAN_ROOT_PATH = os.path.dirname(NUPLAN_PACKAGE_PATH)
    NUPLAN_TUTORIAL_PATH = os.path.join(os.path.dirname(nuplan.__file__), "tutorial")

    def __init__(self):
        super(NuPlanDataManager, self).__init__()
        logger.info("\n \n ############### Start Loading NuPlan Data ############### \n")
        # get original nuplan cfg
        cfg = self._get_nuplan_cfg()
        profiler_name = 'building_simulation'
        common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

        # Build scenario builder
        scenario_builder = build_scenario_builder(cfg=cfg)
        scenario_filter = build_scenario_filter(cfg.scenario_filter)

        # get scenarios from database
        self._nuplan_scenarios = scenario_builder.get_scenarios(scenario_filter, common_builder.worker)

        # filter case according to config
        self.start_case_index = self.engine.global_config["start_case_index"]
        self.case_num = self.engine.global_config["case_num"]
        assert len(self._nuplan_scenarios) >= self.start_case_index + self.case_num, \
            "Number of scenes are not enough, " \
            "\n num nuplan scenarios: {}" \
            "\n start_case_index: {}" \
            "\n case num: {}".format(len(self._nuplan_scenarios), self.start_case_index, self.case_num)
        logger.info("\n \n ############### Finish Loading NuPlan Data ############### \n")

        self._scenario_num = self.case_num
        self._current_scenario_index = None

    @property
    def time_interval(self):
        return self.current_scenario.database_interval

    @property
    def scenario_num(self):
        return self._scenario_num

    @property
    def current_scenario_index(self):
        return self._current_scenario_index

    @property
    def current_scenario(self):
        return self._nuplan_scenarios[self._current_scenario_index]

    def _get_nuplan_cfg(self):
        BASE_CONFIG_PATH = os.path.join(self.NUPLAN_PACKAGE_PATH, "planning", "script")
        simulation_hydra_paths = construct_simulation_hydra_paths(BASE_CONFIG_PATH)

        # Initialize configuration management system
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
        hydra.initialize_config_dir(config_dir=simulation_hydra_paths.config_path)

        SAVE_DIR = tempfile.mkdtemp()
        EGO_CONTROLLER = 'perfect_tracking_controller'  # [log_play_back_controller, perfect_tracking_controller]
        OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, lidar_pc_observation]

        DATASET_PARAMS = self.engine.global_config["DATASET_PARAMS"]

        # Compose the configuration
        cfg = hydra.compose(
            config_name=simulation_hydra_paths.config_name,
            overrides=[
                f'group={SAVE_DIR}',
                'worker=sequential',
                f'ego_controller={EGO_CONTROLLER}',
                f'observation={OBSERVATION}',
                f'hydra.searchpath=[{simulation_hydra_paths.common_dir}, {simulation_hydra_paths.experiment_dir}]',
                'output_dir=${group}/${experiment}',
                *DATASET_PARAMS,

                # TODO: Check which one is correct.
                # Option 1: LQY write:
                f'experiment_name=planner_tutorial',

                # Option 2: planning tutorial:
                # Copied from tutorial
                # f'job_name=planner_tutorial',
                # 'experiment=${experiment_name}/${job_name}/${experiment_time}',
            ]
        )
        return cfg

    def get_case(self, index, force_get_current_case=True):
        assert self.start_case_index <= index < self.start_case_index + self.case_num
        if force_get_current_case:
            assert index == self.random_seed
            return self.current_scenario
        else:
            return self._nuplan_scenarios[index]

    def seed(self, random_seed):
        assert self.start_case_index <= random_seed < self.start_case_index + self.case_num
        super(NuPlanDataManager, self).seed(random_seed)
        self._current_scenario_index = random_seed

    @property
    def current_scenario_length(self):
        return self.current_scenario.get_number_of_iterations()
