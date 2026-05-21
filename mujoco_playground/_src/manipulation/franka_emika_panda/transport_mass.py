# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda.panda import _ARM_JOINTS
from mujoco_playground._src.manipulation.franka_emika_panda import panda_kinematics
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np

#Position of where EE should be moved
TARGET_POS = [0.7, -0.2, 0.0255]
#x,y coords of points that span the area of spawn points
TRIANGLE_POINTS = [
  [0.7, 0.2],
  [0.3, -0.2],
  [0.3, 0.2],
]
#max, min height of EE in spawn point
SPAWN_MIN_HEIGHT = 0.04
SPAWN_MAX_HEIGHT = 0.09


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=150,
      action_repeat=1,
      action_scale=0.04,
      reward_config=config_dict.create(
          scales=config_dict.create(
              r_dist=1.0,
              r_exact=1.0,
              p_vel=-1.0,
              p_neutral=-1.0,
              p_limit=-1.0,
              p_smooth=-1.0,
              # Do not collide the gripper with the floor.
              p_floor_collision=-1.0,
          ),
          r_exact_epsilon=0.003,
      ),
      min_mass=0.0,
      max_mass=3.0,
      impl='jax',
      nconmax=24 * 8192,
      njmax=128,
  )
  return config

def random_rotation_z(key):
  theta = jax.random.uniform(key, minval=-jp.pi, maxval=jp.pi)

  c = jp.cos(theta)
  s = jp.sin(theta)

  return jp.array([
    [c, -s, 0.0],
    [s,  c, 0.0],
    [0.0, 0.0, 1.0],
  ])


class PandaTransportMass(panda.PandaBase):
  """Bring a box (mounted to the Gripper) of random mass to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_mass.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(keyframe="home")
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]
    # Read out the range of the joints of robot
    self.q_min = self.mj_model.jnt_range[:7, 0]
    self.q_max = self.mj_model.jnt_range[:7, 1]
    # Save payload and target IDs
    self._payload_id = self.mj_model.body("payload").id

  def reset(self, rng: jax.Array) -> State:
    rng, rng_mass, rng_spawn = jax.random.split(rng, 3)
    # initialize data
    init_q = (
        jp.array(self._init_q)
    )


    spawn_q = self.sample_spawn_pose(rng_spawn, init_q)


    data = mjx_env.make_data(
        self._mj_model,
        qpos=spawn_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "reached_box": 0.0, "start_pos": spawn_q, "last_action": jp.zeros(7)}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    pre_state = State(data, obs, reward, done, metrics, info)
    # randomize mass
    mass = jax.random.uniform(rng_mass, minval=self._config.min_mass, maxval=self._config.max_mass)
    return self.set_payload_mass(pre_state, mass)

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)
    #Using this method should allow parallel envs with different mass
    mjx_model = state.info["mjx_model"]

    data = mjx_env.step(mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info, ctrl)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=0
    )
    #update info
    info = dict(state.info)
    info["last_action"] = ctrl

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, info)
    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any], action: jax.Array) -> Dict[str, Any]:
    #Rew terms
    r_dist = self._r_dist(data, info)
    r_exact = self._r_exact(data, info)
    #Pen terms
    r_vel = self._r_vel(data, info)
    r_neutral = self._r_neutral(data, info)
    r_limit = self._r_limit(data, info)
    r_smooth = self._r_smooth(action, info)
    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)


    rewards = {
        "r_dist": r_dist,
        "r_exact": r_exact,
        "p_vel": r_vel,
        "p_neutral": r_neutral,
        "p_limit": r_limit,
        "p_smooth": r_smooth,
        "p_floor_collision": no_floor_collision,
    }
    print(rewards)
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
    ])

    return obs

  def _post_init(self, keyframe: str):
    all_joints = _ARM_JOINTS
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in _ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in all_joints
    ])
    self._gripper_site = self._mj_model.site("gripper").id
    self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
    self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
    self._hand_geom = self._mj_model.geom("hand_capsule").id
    self._target_id = self._mj_model.body("target").id
    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T
    #save target qpos keyframe
    self._target_q = self._mj_model.keyframe("target").qpos

  def set_payload_mass(self, state: State, mass):
    """Takes a state and sets the mass of the payload to the given mass.
    Returns a new state with the correct mass.
    Should only be used directly after reset, because metrics might be wonky otherwise..."""
    self._mj_model.body("payload").mass = mass
    payload_id = self._mj_model.body("payload").id
    mjx_model = self._mjx_model.replace(
      body_mass=self._mjx_model.body_mass.at[payload_id].set(mass)
    )
    info = dict(state.info)
    info["mjx_model"] = mjx_model
    info["payload_mass"] = mass
    return State(state.data, state.obs, state.reward, state.done, state.metrics, info)

  def sample_spawn_pose(self, key, q_actual):
    def sample_point_in_triangle(key):
      A = jp.array(TRIANGLE_POINTS[0])
      B = jp.array(TRIANGLE_POINTS[1])
      C = jp.array(TRIANGLE_POINTS[2])

      u,v = jax.random.uniform(key, (2,))
      #Mirror if outside the triangle
      u, v = jp.where(u+v > 1.0, 1.0-u, u), jp.where(u+v > 1.0, 1.0-v, v)

      xy = A + u*(B-A) + v*(C-A)
      return xy
    
    def make_ee_pose(key, xy, z):
      T = jp.eye(4)
      T = T.at[:3, 3].set(jp.array([xy[0], xy[1], z]))
      R = jp.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
      ])
      R_rot = random_rotation_z(key)
      R = R_rot @ R
      T = T.at[:3, :3].set(R)
      return T
    
    def try_spawn_pose(key, q_actual):
      k1, k2, k3 = jax.random.split(key, 3)
    

      xy = sample_point_in_triangle(k1)
      z = jax.random.uniform(k2, minval=SPAWN_MIN_HEIGHT, maxval=SPAWN_MAX_HEIGHT)
      T = make_ee_pose(k3, xy, z)

      q7 = q_actual[6]
      q = panda_kinematics.compute_franka_ik(T, q7=q7, q_actual=q_actual)
      return q
    
    def cond(carry):
      _, q = carry
      return ~jp.all(jp.isfinite(q))
    
    def body(carry):
      key, _ = carry
      key, subkey = jax.random.split(key)
      q = try_spawn_pose(subkey, q_actual)
      return key, q
    
    init = (key, jp.full_like(q_actual, jp.nan))

    _, q = jax.lax.while_loop(cond, body, init)

    return q
  
  def _r_dist(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    d_target = jp.square(data.qpos - self._target_q).sum()
    return 1 / (1+d_target)
  
  def _r_exact(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    norm = jp.linalg.norm(data.qpos - self._target_q)
    close = norm < self._config.reward_config.r_exact_epsilon
    q_dot_squared = jp.sum(jp.square(data.qvel))
    lam = self._config.reward_config.scales["r_exact"]
    val = 1.0 + 1.0 / (lam * (1.0 + 100.0 * q_dot_squared)) #IS correct, because in step() it will be multiplied with self._config.reward_config.scales["r_exact"]
    return jp.where(close, val, 0.0)
  
  def _r_vel(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    return jp.square(data.qvel).sum()
  
  def _r_neutral(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    return jp.linalg.norm(data.qpos - info["start_pos"])
  
  def _r_limit(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    q_dif_min = data.qpos - self.q_min
    q_dif_max = data.qpos - self.q_max
    q_dif = jp.minimum(q_dif_min, q_dif_max)
    return jp.exp(-30*jp.square(q_dif)).sum()
  
  def _r_smooth(self, action: jax.Array, info: Dict[str, Any]) -> float:
    return jp.linalg.norm(action - info["last_action"])

  #def get_target_qpos(slef, key, q_actual, n_tries=128):
  #  target_payload_pos = jp.array(TARGET_POS)
  #  target_payload_pos = target_payload_pos - jp.array([0.0,0.0,0.0084])
  #  keys = jax.random.split(key, n_tries)

  #  def make_(target_pos, key):
  #    k_rot = key
  #    base_r = jp.array([
  #      [1.0,0.0,0.0],
  #      [0.0,-1.0,0.0],
  #      [0.0,0.0,-1.0],
  #    ])
  #    Rz = jp.array([
  #      [jp.cos(jp.pi/2), -jp.sin(jp.pi/2), 0.0],
  #      [jp.sin(jp.pi/2), jp.cos(jp.pi/2), 0.0],
  #      [0.0,0.0,1.0]
  #    ])
  #    #Rz = random_rotation_z(k_rot)
  #    Rz = Rz @ base_r
  #    #Rz = base_r


  #     T = jp.eye(4)
  #    T = T.at[:3, 3].set(target_pos)
  #    T = T.at[:3, :3].set(Rz)
  #    return T
  #  
  #  def try_one(k):
  #    k_rot, k_q7 = jax.random.split(k)
  #    T = make_(target_payload_pos, k_rot)
  #    q7 = jax.random.uniform(
  #      k_q7,
  #      minval=-2.8973, maxval=2.8973
  #    )
  #    q = panda_kinematics.compute_franka_ik(T, q7, q_actual)
  #    valid = jp.all(jp.isfinite(q))
  #    return q, valid
  #  
  #  qs, valids = jax.vmap(try_one)(keys)
  #  idx = jp.argmax(valids)
  #  return jp.where(jp.any(valids), qs[idx], jp.full_like(q_actual, jp.nan))

  


#import matplotlib.pyplot as plt
#env = PandaTransportMass()
#state = env.reset(jax.random.PRNGKey(2380))

#traj = []
#traj.append(state)

#instructions = [
#   jp.array([50, 50, 0, -100, 0, 0, 0]),
#   jp.array([50, 50, 0, -100, 0, 0, 0]),
#   jp.array([50, 50, 0, -100, 0, 0, 0]),
#   jp.array([50, 50, 0, -100, 0, 0, 0]),
#   jp.array([50, 50, 0, 0, 0, 10, 0]),
#   jp.array([50, 50, 0, 0, 0, 10, 0]),
#   jp.array([100, 0, 0, 0, 0, 10, 0]),
#]

#for ins in instructions:
#  state = env.step(state, ins)
#  print(state.reward)
#  traj.append(state)

#images = env.render(traj, height=480, width=640)


#for im in images:
#    plt.imshow(im)
#    plt.show()