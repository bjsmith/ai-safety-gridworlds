# Copyright 2022 Roland Pihlakas. All Rights Reserved.
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

import itertools

# Dependency imports
import numpy as np


class mo_reward(object):

  def __init__(self, reward_dimensions_dict, immutable=True):

    self._reward_dimensions_dict = reward_dimensions_dict
    self._immutable = immutable


  def copy(self):

    dict_clone = dict(self._reward_dimensions_dict) # clone
    return mo_reward(dict_clone, immutable=False)


  def get_enabled_reward_dimension_keys(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      return [None]

    else: # if enabled_mo_reward_dimensions is not None:

      # each reward may contain more than one enabled dimension
      keys_per_reward = [{ key for key, unit_value in reward._reward_dimensions_dict.items() if unit_value != 0 }
                                                        for reward in enabled_mo_reward_dimensions]
      # enabled_reward_dimension_keys = set.union(*keys_per_reward)  # this does not preserve the order of the keys
      enabled_reward_dimension_keys = dict.fromkeys(itertools.chain.from_iterable(keys_per_reward)).keys()  # this preserves the order of the keys
      return list(enabled_reward_dimension_keys)


  def tolist(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      reward_values = self._reward_dimensions_dict.values()
      return sum(reward_values)

    else: # if enabled_mo_reward_dimensions is not None:

      enabled_reward_dimension_keys = self.get_enabled_reward_dimension_keys(enabled_mo_reward_dimensions)

      for key in self._reward_dimensions_dict.keys():
        if key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward" % key)

      result = [0] * len(enabled_reward_dimension_keys)
      for dimension_index, enabled_reward_dimension_key in enumerate(enabled_reward_dimension_keys):
        result[dimension_index] = self._reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def tofull(self, enabled_mo_reward_dimensions):

    if enabled_mo_reward_dimensions is None:

      reward_values = self._reward_dimensions_dict.values()
      return {None: sum(reward_values)}

    else: # if enabled_mo_reward_dimensions is not None:

      enabled_reward_dimension_keys = self.get_enabled_reward_dimension_keys(enabled_mo_reward_dimensions)

      for key in self._reward_dimensions_dict.keys():
        if key not in enabled_reward_dimension_keys:
          raise ValueError("Reward %s is not enabled but is still included in mo_reward" % key)

      result = {}
      for enabled_reward_dimension_key in enabled_reward_dimension_keys:
        result[enabled_reward_dimension_key] = self._reward_dimensions_dict.get(enabled_reward_dimension_key, 0)
      return result


  def __str__(self): # tostring

    return str(self._reward_dimensions_dict)


  def __repr__(self): # tostring

    return "<" + repr(self._reward_dimensions_dict) + ">"


  def __add__(self, other):

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) + other_value
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__add__, expecting a scalar or mo_reward")


  def __iadd__(self, other):  # in-place add

    if self._immutable:
      return self.__add__(other)

    if np.isscalar(other):
      for key, value in self._reward_dimensions_dict.items():
        self._reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        self._reward_dimensions_dict[other_key] = self._reward_dimensions_dict.get(other_key, 0) + other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__iadd__, expecting a scalar or mo_reward")

    return self


  def __radd__(self, other):  # reflected add (the order of operands is exchanged)

    return self + other


  def __sub__(self, other):

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: value + other for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = result_dict.get(other_key, 0) - other_value
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__sub__, expecting a scalar or mo_reward")


  def __isub__(self, other):  # in-place sub

    if self._immutable:
      return self.__sub__(other)

    if np.isscalar(other):
      for key, value in self._reward_dimensions_dict.items():
        self._reward_dimensions_dict[key] = value + other

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        self._reward_dimensions_dict[other_key] = self._reward_dimensions_dict.get(other_key, 0) - other_value

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__isub__, expecting a scalar or mo_reward")

    return self


  def __rsub__(self, other):  # reflected sub (the order of operands is exchanged)

    result_dict = dict(self._reward_dimensions_dict)  # clone

    if np.isscalar(other):
      return mo_reward({ key: other - value for key, value in self._reward_dimensions_dict.items() }, immutable=False)

    elif isinstance(other, mo_reward):
      for other_key, other_value in other._reward_dimensions_dict.items():
        result_dict[other_key] = other_value - result_dict.get(other_key, 0)
      return mo_reward(result_dict, immutable=False)

    else:
      raise NotImplementedError("Unknown value type provided for mo_reward.__rsub__, expecting a scalar or mo_reward")


  def __neg__(self):  # unary -

    return mo_reward({ key: -value for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __mul__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__mul__, expecting a scalar")

    return mo_reward({ key: value * other for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __imul__(self, other):  #in-place mul

    if self._immutable:
      return self.__mul__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__imul__, expecting a scalar")

    for key, value in self._reward_dimensions_dict.items():
      self._reward_dimensions_dict[key] = value * other

    return self


  def __rmul__(self, other):  # reflected mul (the order of operands is exchanged)

    return self * other


  def __truediv__(self, other):

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__truediv__, expecting a scalar")

    return mo_reward({ key: value / other for key, value in self._reward_dimensions_dict.items() }, immutable=False)


  def __itruediv__(self, other):  #in-place div

    if self._immutable:
      return self.__truediv__(other)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__itruediv__, expecting a scalar")

    for key, value in self._reward_dimensions_dict.items():
      self._reward_dimensions_dict[key] = value / other

    return self


  def __rtruediv__(self, other):  # reflected div (the order of operands is exchanged)

    if not np.isscalar(other):
      raise NotImplementedError("Unknown value type provided for mo_reward.__rtruediv__, expecting a scalar")

    return mo_reward({ key: other / value for key, value in self._reward_dimensions_dict.items() }, immutable=False)


