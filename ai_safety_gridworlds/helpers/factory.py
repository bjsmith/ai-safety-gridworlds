# Copyright 2022 Roland Pihlakas. https://github.com/levitation-opensource/multiobjective-ai-safety-gridworlds
# Copyright 2018 The AI Safety Gridworlds Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Module containing factory class to instantiate all pycolab environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ai_safety_gridworlds.environments.absent_supervisor import AbsentSupervisorEnvironment
from ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment
from ai_safety_gridworlds.environments.boat_race_ex import BoatRaceEnvironmentEx
from ai_safety_gridworlds.environments.conveyor_belt import ConveyorBeltEnvironment
from ai_safety_gridworlds.environments.conveyor_belt_ex import ConveyorBeltEnvironmentEx
from ai_safety_gridworlds.environments.distributional_shift import DistributionalShiftEnvironment
from ai_safety_gridworlds.environments.friend_foe import FriendFoeEnvironment

from ai_safety_gridworlds.environments.island_navigation import IslandNavigationEnvironment
from ai_safety_gridworlds.environments.island_navigation_ex import IslandNavigationEnvironmentEx
from ai_safety_gridworlds.experiments import food_drink_unbounded
from ai_safety_gridworlds.experiments import food_drink_rolf
from ai_safety_gridworlds.experiments import food_drink_rolf_gold
from ai_safety_gridworlds.experiments import food_drink_bounded
from ai_safety_gridworlds.experiments import food_drink_bounded_gold
from ai_safety_gridworlds.experiments import food_drink_bounded_gold_silver
from ai_safety_gridworlds.experiments import food_drink_bounded_death
from ai_safety_gridworlds.experiments import food_drink_bounded_death_gold
from ai_safety_gridworlds.experiments import food_drink_bounded_death_gold_silver

from ai_safety_gridworlds.environments.rocks_diamonds import RocksDiamondsEnvironment
from ai_safety_gridworlds.environments.safe_interruptibility import SafeInterruptibilityEnvironment
from ai_safety_gridworlds.environments.safe_interruptibility_ex import SafeInterruptibilityEnvironmentEx
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment
from ai_safety_gridworlds.environments.tomato_crmdp import TomatoCRMDPEnvironment
from ai_safety_gridworlds.environments.whisky_gold import WhiskyOrGoldEnvironment


_environment_classes = {
    'boat_race': BoatRaceEnvironment,
    'boat_race_ex': BoatRaceEnvironmentEx,
    'conveyor_belt': ConveyorBeltEnvironment,
    'conveyor_belt_ex': ConveyorBeltEnvironmentEx,
    'distributional_shift': DistributionalShiftEnvironment,
    'friend_foe': FriendFoeEnvironment,

    'island_navigation': IslandNavigationEnvironment,
    'island_navigation_ex': IslandNavigationEnvironmentEx,
    'food_drink_unbounded': food_drink_unbounded.IslandNavigationEnvironmentExExperiment,
    'food_drink_rolf': food_drink_rolf.IslandNavigationEnvironmentExExperiment,
    'food_drink_rolf_gold': food_drink_rolf_gold.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded': food_drink_bounded.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_gold': food_drink_bounded_gold.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_gold_silver': food_drink_bounded_gold_silver.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_death': food_drink_bounded_death.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_death_gold': food_drink_bounded_death_gold.IslandNavigationEnvironmentExExperiment,
    'food_drink_bounded_death_gold_silver': food_drink_bounded_death_gold_silver.IslandNavigationEnvironmentExExperiment,

    'rocks_diamonds': RocksDiamondsEnvironment,
    'safe_interruptibility': SafeInterruptibilityEnvironment,
    'safe_interruptibility_ex': SafeInterruptibilityEnvironmentEx,
    'side_effects_sokoban': SideEffectsSokobanEnvironment,
    'tomato_watering': TomatoWateringEnvironment,
    'tomato_crmdp': TomatoCRMDPEnvironment,
    'absent_supervisor': AbsentSupervisorEnvironment,
    'whisky_gold': WhiskyOrGoldEnvironment,
}


def get_environment_obj(name, *args, **kwargs):
  """Instantiate a pycolab environment by name.

  Args:
    name: Name of the pycolab environment.
    *args: Arguments for the environment class constructor.
    **kwargs: Keyword arguments for the environment class constructor.

  Returns:
    A new environment class instance.
  """
  environment_class = _environment_classes.get(name.lower(), None)

  if environment_class:
    return environment_class(*args, **kwargs)
  raise NotImplementedError(
      'The requested environment is not available.')


register_with_gym_done = False
def register_with_gym():
  global register_with_gym_done


  if register_with_gym_done:  # avoid warnings caused by duplicate registrations
    return
  register_with_gym_done = True


  from gym.envs.registration import register

  env_list = _environment_classes.keys()
  

  # adapted from safe_grid_gym\__init__.py
  def to_gym_id(env_name):
      result = []
      nextUpper = True
      for char in env_name:
          if nextUpper:
              result.append(char.upper())
              nextUpper = False
          elif char == "_":
              nextUpper = True
          else:
              result.append(char)
      return "".join(result)

  # adapted from safe_grid_gym\envs\__init__.py
  def get_id(env_name):
      return "ai_safety_gridworlds-" + env_name + "-v0"


  for env_name in env_list:

      # adapted from safe_grid_gym\__init__.py
      gym_id_prefix = to_gym_id(str(env_name))
      if gym_id_prefix == "ConveyorBelt":
          for variant in ['vase', 'sushi', 'sushi_goal', 'sushi_goal2']:
              register(
                  id=to_gym_id(str(variant)) + "-v0",
                  entry_point="ai_safety_gridworlds.helpers.gridworld_gym_env:GridworldGymEnv",
                  kwargs={"env_name": env_name, "variant": variant},
              )
      else:
          register(
              id=gym_id_prefix + "-v0",
              entry_point="ai_safety_gridworlds.helpers.gridworld_gym_env:GridworldGymEnv",
              kwargs={"env_name": env_name},
          )

      # adapted from safe_grid_gym\envs\__init__.py
      register(
          id=get_id(env_name),
          entry_point="ai_safety_gridworlds.helpers.gridworld_gym_env:GridworldGymEnv",
          kwargs={"env_name": env_name, "pause": 0.2},
      )
