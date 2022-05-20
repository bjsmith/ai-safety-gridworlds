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
"""Helpers for creating safety environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from ai_safety_gridworlds.environments.shared.termination_reason_enum import TerminationReason
from ai_safety_gridworlds.environments.shared.safety_game import SafetyEnvironment, ACTUAL_ACTIONS, TERMINATION_REASON, EXTRA_OBSERVATIONS

import numpy as np



class mo_reward(object):

  def __init__(self, reward_dimensions_dict):

    self.reward_dimensions_dict = reward_dimensions_dict


  def get_enabled_reward_dimension_keys(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      return [None]

    else: # if enabled_mo_reward_dimensions is not None:

      # each reward may contain more than one enabled dimension
      keys_per_reward = [{ key for key, unit_value in reward.reward_dimensions_dict.items() if unit_value != 0 }
                                                        for reward in enabled_mo_reward_dimensions]
      enabled_reward_dimension_keys = set.union(*keys_per_reward)
      return enabled_reward_dimension_keys


  def tolist(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      reward_values = self.reward_dimensions_dict.values()
      return sum(reward_values)

    else: # if enabled_mo_reward_dimensions is not None:

      enabled_reward_dimension_keys = self.get_enabled_reward_dimension_keys(enabled_mo_reward_dimensions)

      for key in self.reward_dimensions_dict.keys():
        if key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward" % key)

      result = [0] * len(enabled_reward_dimension_keys)
      for dimension_index, enabled_reward_dimension_key in enumerate(enabled_reward_dimension_keys):
        result[dimension_index] = self.reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def tofull(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      reward_values = self.reward_dimensions_dict.values()
      return {None: sum(reward_values)}

    else: # if enabled_mo_reward_dimensions is not None:

      enabled_reward_dimension_keys = self.get_enabled_reward_dimension_keys(enabled_mo_reward_dimensions)

      for key in self.reward_dimensions_dict.keys():
        if key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward" % key)

      result = {}
      for enabled_reward_dimension_key in enabled_reward_dimension_keys:
        result[enabled_reward_dimension_key] = self.reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def __str__(self): # tostring

    return str(self.reward_dimensions_dict)


  def __repr__(self): # tostring

    return "<" + repr(self.reward_dimensions_dict) + ">"


  def __add__(self, other):

    result_dict = dict(reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self.reward_dimensions_dict.items() })

    elif isinstance(other, mo_reward):
      for other_key, other_value in other.reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) + other_value
      return mo_reward(result_dict)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__add__, expecting a scalar or mo_reward")


  def __iadd__(self, other):  # in-place add

    if np.isscalar(other):
      for key, value in self.reward_dimensions_dict.items():
        self.reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other.reward_dimensions_dict.items():
        self.reward_dimensions_dict[other_key] = self.reward_dimensions_dict.get(other_key, 0) + other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__iadd__, expecting a scalar or mo_reward")

    return self


  def __radd__(self, other):  # reflected add (the order of operands is exchanged)

    return self + other


  def __sub__(self, other):

    result_dict = dict(reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self.reward_dimensions_dict.items() })

    elif isinstance(other, mo_reward):
      for other_key, other_value in other.reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) - other_value
      return mo_reward(result_dict)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__sub__, expecting a scalar or mo_reward")


  def __isub__(self, other):  # in-place sub

    if np.isscalar(other):
      for key, value in self.reward_dimensions_dict.items():
        self.reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other.reward_dimensions_dict.items():
        self.reward_dimensions_dict[other_key] = self.reward_dimensions_dict.get(other_key, 0) - other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__isub__, expecting a scalar or mo_reward")

    return self


  def __rsub__(self, other):  # reflected sub (the order of operands is exchanged)

    result_dict = dict(reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: other - value for key, value in self.reward_dimensions_dict.items() })

    elif isinstance(other, mo_reward):
      for other_key, other_value in other.reward_dimensions_dict.items():
        result_dict[other_key] = other_value - result_dict.get(other_key, 0)
      return mo_reward(result_dict)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__rsub__, expecting a scalar or mo_reward")


  def __mul__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__mul__, expecting a scalar")

    return mo_reward({ key: value * other for key, value in self.reward_dimensions_dict.items() })


  def __imul__(self, other):  #in-place mul

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__imul__, expecting a scalar")

    for key, value in self.reward_dimensions_dict.items():
      self.reward_dimensions_dict[key] = value * other

    return self


  def __rmul__(self, other):  # reflected mul (the order of operands is exchanged)

    return self * other


  def __truediv__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__truediv__, expecting a scalar")

    return mo_reward({ key: value / other for key, value in self.reward_dimensions_dict.items() })


  def __itruediv__(self, other):  #in-place div

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__itruediv__, expecting a scalar")

    for key, value in self.reward_dimensions_dict.items():
      self.reward_dimensions_dict[key] = value / other

    return self


  def __rtruediv__(self, other):  # reflected div (the order of operands is exchanged)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__rtruediv__, expecting a scalar")

    return mo_reward({ key: other / value for key, value in self.reward_dimensions_dict.items() })



class SafetyEnvironmentMo(SafetyEnvironment):
  """Base class for multi-objective safety gridworld environments.

  Environments implementing this base class initialize the Python environment
  API v2 and serve as a layer in which we can put various modifications of
  pycolab inputs and outputs, such as *additional information* passed
  from/to the environment that does not fit in the traditional observation
  channel. It also allows for overwriting of default methods such as step() and
  reset().

  Each new environment must implement a subclass of this class, and at the very
  least call the __init__ method of this class with corresponding parameters, to
  instantiate the python environment API around the pycolab game.
  """

  def __init__(self, enabled_mo_reward_dimensions, 
               #game_factory,
               #game_bg_colours,
               #game_fg_colours,
               #actions=None,
               #value_mapping=None,
               #environment_data=None,
               #repainter=None,
               #max_iterations=100,
               *args, **kwargs):
    """Initialize a Python v2 environment for a pycolab game factory.

    Args:
      enabled_mo_reward_dimensions: list of multi-objective rewards being used in current 
        map. Providing this list enables reducing the dimensionality of the 
        reward vector in such a way that unused reward dimensions are left out. 
        If set to None then the multi-objective rewards are disabled: the 
        rewards are then scalarised before returning to the agent.
      game_factory: a function that returns a new pycolab `Engine`
        instance corresponding to the game being played.
      game_bg_colours: a dict mapping game characters to background RGB colours.
      game_fg_colours: a dict mapping game characters to foreground RGB colours.
      actions: a tuple of ints, indicating an inclusive range of actions the
        agent can take. Defaults to DEFAULT_ACTION_SET range.
      value_mapping: a dictionary mapping characters from the game ascii map
        into floats. Used to control how the agent sees the game ascii map, e.g.
        if we are not making a difference between environment background and
        walls in terms of values the agent sees for those blocks, we can map
        them to the same value. Defaults to mapping characters to their ascii
        codes.
      environment_data: dictionary of data that is passed to the pycolab
        environment implementation and is used as a shared object that allows
        each wrapper to communicate with their environment. This object can hold
        additional information about the state of the environment that can even
        persists through episodes, but some particular keys are erased at each
        new episode.
      repainter: a callable that converts `rendering.Observation`s to different
        `rendering.Observation`s, or None if no such conversion is required.
        This facility is normally used to change the characters used to depict
        certain game elements, and a `rendering.ObservationCharacterRepainter`
        object is a convenient way to accomplish this conversion. For more
        information, see pycolab's `rendering.py`.
      max_iterations: the maximum number of steps for one episode.
    """

    self.enabled_mo_reward_dimensions = enabled_mo_reward_dimensions

    super(SafetyEnvironmentMo, self).__init__(*args, **kwargs)

    # parent class safety_game.SafetyEnvironment sets default_reward=0
    self._default_reward = mo_reward({})


  #def _calculate_overall_performance(self):
  #  """Calculates the agent performance across all the episodes.

  #  By default, the method will return the average across all episodes.
  #  You should override this method if you want to implement some other way of
  #  calculating the overall performance.

  #  Returns:
  #    A float value summarizing the performance of the agent.
  #  """
  #  return sum(self._episodic_performances) / len(self._episodic_performances)

  #def _get_hidden_reward(self, default_reward=0):
  #  """Extract the hidden reward from the plot of the current episode."""
  #  return self.current_game.the_plot.get(HIDDEN_REWARD, default_reward)


  # adapted from safety_game.py SafetyEnvironment._process_timestep(self, timestep)
  def _process_timestep(self, timestep):
    """Do timestep preprocessing before sending it to the agent.

    This method stores the cumulative return and makes sure that the
    `environment_data` is included in the observation.

    If you are overriding this method, make sure to call `super()` to include
    this code.

    Args:
      timestep: instance of environment.TimeStep

    Returns:
      Preprocessed timestep.
    """

    # Reset the cumulative episode reward.
    if timestep.first():
      self._episode_return = mo_reward({})    # CHANGED for multi-objective rewards
      self._clear_hidden_reward()
      # Clear the keys in environment data from the previous episode.
      for key in self._keys_to_clear:
        self._environment_data.pop(key, None)

    # Add the timestep reward for internal wrapper calculations.
    if timestep.reward:
      self._episode_return += timestep.reward
    extra_observations = self._get_agent_extra_observations()

    if ACTUAL_ACTIONS in self._environment_data:
      extra_observations[ACTUAL_ACTIONS] = (
          self._environment_data[ACTUAL_ACTIONS])

    if timestep.last():
      # Include the termination reason for the episode if missing.
      if TERMINATION_REASON not in self._environment_data:
        self._environment_data[TERMINATION_REASON] = TerminationReason.MAX_STEPS

      extra_observations[TERMINATION_REASON] = (
          self._environment_data[TERMINATION_REASON])

    timestep.observation[EXTRA_OBSERVATIONS] = extra_observations

    # Calculate performance metric if the episode has finished.
    if timestep.last():
      self._calculate_episode_performance(timestep)


    # CHANGED: add conversion of mo_reward to a list
    if timestep.reward is not None:
      timestep = timestep._replace(reward=timestep.reward.tolist(self.enabled_mo_reward_dimensions))
    else: # NB! do not return None since safe_grid_gym would convert that to scalar 0
      timestep = timestep._replace(reward=mo_reward({}).tolist(self.enabled_mo_reward_dimensions))


    return timestep

