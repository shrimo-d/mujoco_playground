"""Push a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx, mju_euler2Quat, mju_mulQuat
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda.panda import _ARM_JOINTS
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np

INIT_POS = [0.5, 0, 0]

ENDEFFECTOR_HEIGHT = 0.07 #m = 70 mm

MIN_SIZE = 0.005 #MuJoCo uses half sizes
MAX_SIZE = 0.05

MIN_MASS = 0.01
MAX_MASS = 20
MIN_SLIDE = 0.5
MAX_SLIDE = 2.0
MIN_ROLL = 0.01
MAX_ROLL = 0.1
MIN_SPIN = 0.001
MAX_SPIN = 0.01

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
              # Gripper goes to the box.
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.3,
          )
      ),
      impl='jax',
      nconmax=24 * 8192,
      njmax=128,
  )
  return config


class PandaPush(panda.PandaBase):
  """Bring a box to a target."""
  geometries = ["box", "capsule", "sphere"]

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
      domain_randomization: bool = True,
      geometry: str = "box",
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_push.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name=geometry, keyframe="home") #this method overides the _post_init of the base panda env.
    self._sample_orientation = sample_orientation
    self._domain_randomization = domain_randomization

    # Contact sensor IDs.
    self._floor_endeffector_found_sensor = [
      self._mj_model.sensor(f"endeffector_{name}_floor_found").id
      for name in ["head", "stick"]
    ]

  def _randomize_domain(self, geom: str, rng: jax.random.PRNGKey):
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if geom=="box":
        #Move other objects far away
        self._mj_model.body("capsule").pos = [100, 100, 0]
        self._mj_model.body("sphere").pos = [101, 100, 0]
        self._mj_model.body("capsule_target").pos = [102, 100, 0]
        self._mj_model.body("sphere_target").pos = [103, 100, 0]
        #Randomize Size (because its geometry based)
        size = jax.random.uniform(rng, (3,), minval=MIN_SIZE, maxval=MAX_SIZE)
        height_offset = size[-1]
        #Randomly rotate around z-axis

    elif geom=="capsule":
        #Move other objects far away
        self._mj_model.body("box").pos = [100, 100, 0]
        self._mj_model.body("sphere").pos = [101, 100, 0]
        self._mj_model.body("box_target").pos = [102, 100, 0]
        self._mj_model.body("sphere_target").pos = [103, 100, 0]
        #Randomize Size (because its geometry based)
        size = jax.random.uniform(rng, (2,), minval=MIN_SIZE, maxval=MAX_SIZE)
        #Flip capsule on the side
        quat = jp.array([jp.sqrt(0.5), jp.sqrt(0.5), 0.0, 0.0], dtype=float)
        height_offset = size[0] #because capsule is defined by giving radius followed by cylinder halfheight

    elif geom=="sphere":
        #Move other objects far away
        self._mj_model.body("box").pos = [100, 100, 0]
        self._mj_model.body("capsule").pos = [101, 100, 0]
        self._mj_model.body("box_target").pos = [102, 100, 0]
        self._mj_model.body("capsule_target").pos = [103, 100, 0]
        #Randomize Size (because its geometry based)
        size = jax.random.uniform(rng, (1,), minval=MIN_SIZE, maxval=MAX_SIZE)
        height_offset = size[0]

    #Set size
    self._mj_model.geom(geom).size = size
    self._mj_model.geom(f"{geom}_target").size = size
    #Randomize Mass
    self._mj_model.body(geom).mass = jax.random.uniform(rng, (1,), minval=MIN_MASS, maxval=MAX_MASS)
    #Randomize Friction
    self._mj_model.geom(geom).friction = jax.random.uniform(rng, 
                                                            (3,), 
                                                            minval=jp.array([MIN_SLIDE, MIN_ROLL, MIN_SPIN]),
                                                            maxval=jp.array([MAX_SLIDE, MAX_ROLL, MAX_SPIN]))
    #Randomly rotate around z-axis
    z_quat = np.zeros((4,1))
    z_angle = jax.random.uniform(rng, (3,), minval=jp.array([0, 0, -np.pi]), maxval=jp.array([0,0,np.pi]))
    mju_euler2Quat(quat=z_quat, euler=np.array(z_angle), seq="XYZ")
    mju_mulQuat(quat, z_quat, quat)  
    return height_offset, jp.array(quat)

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)
    height_offset = 0
    quat = jp.array([1,0,0,0], dtype=float)

    if self._domain_randomization:
        geom_idx = jax.random.choice(rng_box, len(self.geometries))
        geom = self.geometries[geom_idx]
        self._set_obj_geom(geom)
        height_offset, quat = self._randomize_domain(geom, rng_box)

    # intialize object position
    obj_pos = (
        jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.0]),
            maxval=jp.array([0.2, 0.2, 0.0]),
        )
        + self._init_obj_pos
        + jp.array([0,0,height_offset])
    )

    # initialize target position
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.0]),
            maxval=jp.array([0.2, 0.2, 0.0]),
        )
        + self._init_obj_pos
        + jp.array([0,0,height_offset])
    )

    # initialize data and set obj to pos
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(obj_pos)
    )
    # also rotate according to quat
    init_q = init_q.at[self._obj_qposadr+3 : self._obj_qposadr + 7].set(quat)
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # set target mocap position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(quat),
    )
    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    #raw_rewards = self._get_reward(data, state.info)
    #rewards = {
    #    k: v * self._config.reward_config.scales[k]
    #    for k, v in raw_rewards.items()
    #}
    #reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    reward = 1
    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    #state.metrics.update(
    #    **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    #)

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    pos_err = jp.linalg.norm(target_pos - box_pos)
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])

    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards
  
  def _get_mask(self, shape: tuple, qvel: bool = False):
    """Creates a mask for the self.qpos and self.qvel array to exclude the unused geometries
    Args:
    shape: tuple -> The shape of qpos and qvel
    qvel: bool -> Whether or not mask for qvel is wanted
    Returns:
    mask: jp.array -> mask of the indizes"""
    offset = 6 if qvel else 7
    capsule_mod = self._capsule_qposadr-1 if qvel else self._capsule_qposadr #in qvel index starts 1 earlier, as it is the second geom
    sphere_mod = self._sphere_qposadr-2 if qvel else self._sphere_qposadr #in qvel index starts 2 earlier, as it is the third geom
    if self._obj_body == self._box_body:
        idx = list(range(capsule_mod, capsule_mod+offset))
        idx.extend(list(range(sphere_mod, sphere_mod+offset)))
    elif self._obj_body == self._capsule_body:
        idx = list(range(self._box_qposadr, self._box_qposadr+offset))
        idx.extend(list(range(sphere_mod, sphere_mod+offset)))
    elif self._obj_body == self._sphere_body:
        idx = list(range(self._box_qposadr, self._box_qposadr+offset))
        idx.extend(list(range(capsule_mod, capsule_mod+offset)))
    else:
       raise Exception("self._obj_body does not contain id of legal object")
    exclude = jp.array(idx)
    mask = jp.ones(shape, dtype=bool).at[exclude].set(False)
    return mask
  
  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    print(data.xpos[self._endeffector_geom])
    for id in self._floor_endeffector_found_sensor:
       if data.sensordata[id] > 0:
          print(f"{id} touched the floor!")
    #gripper_pos = data.site_xpos[self._gripper_site] #probably irrelevant for push task, as gripper is assumed fixed?
    #gripper_mat = data.site_xmat[self._gripper_site].ravel() #probably also irrelevant?
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    #Mask out all unused objects from qpos and qvel
    qpos_mask = self._get_mask(data.qpos.shape)
    qvel_mask = self._get_mask(data.qvel.shape, qvel=True)

    obs = jp.concatenate([
        data.qpos[qpos_mask],
        data.qvel[qvel_mask],
        #gripper_pos,
        #gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        #data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr],
    ])

    return obs
  
  def _post_init(self, obj_name: str, keyframe: str):
    self._robot_arm_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in _ARM_JOINTS
    ])
    self._robot_qposadr = np.array([
        self._mj_model.jnt_qposadr[self._mj_model.joint(j).id]
        for j in _ARM_JOINTS
    ])
    #Init endeffector id and qposadr
    self._endeffector_geom = self._mj_model.body("endeffector").id
    self._endeffector_qposadr = self._mj_model.jnt_qposadr[self._mj_model.body("endeffector").jntadr[0]]

    #Init body ids and qposadrs for all geometries
    self._box_body = self._mj_model.body("box").id
    self._box_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("box").jntadr[0]
    ]
    self._capsule_body = self._mj_model.body("capsule").id
    self._capsule_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("capsule").jntadr[0]
    ]
    self._sphere_body = self._mj_model.body("sphere").id
    self._sphere_qposadr = self._mj_model.jnt_qposadr[
        self._mj_model.body("sphere").jntadr[0]
    ]
    self._box_mocap = self._mj_model.body("box_target").mocapid
    self._capsule_mocap = self._mj_model.body("capsule_target").mocapid
    self._sphere_mocap = self._mj_model.body("sphere_target").mocapid

    #Set obj_name as default object and set mocap target to the corresponding geom
    self._set_obj_geom(obj_name)

    self._floor_geom = self._mj_model.geom("floor").id
    self._init_q = self._mj_model.keyframe(keyframe).qpos
    self._init_obj_pos = jp.array(
        INIT_POS,
        dtype=jp.float32,
    )
    self._init_ctrl = self._mj_model.keyframe(keyframe).ctrl
    self._lowers, self._uppers = self._mj_model.actuator_ctrlrange.T

  def _set_obj_geom(self, obj_name: str):
    if obj_name not in self.geometries:
      raise Exception(f"No geometry with name '{obj_name}' in list of supported geometries: {self.geometries}")
    elif obj_name == self.geometries[0]:
      self._obj_body = self._box_body
      self._obj_qposadr = self._box_qposadr
      self._mocap_target = self._box_mocap
    elif obj_name == self.geometries[1]:
      self._obj_body = self._capsule_body
      self._obj_qposadr = self._capsule_qposadr
      self._mocap_target = self._capsule_mocap
    elif obj_name == self.geometries[2]:
      self._obj_body = self._sphere_body
      self._obj_qposadr = self._sphere_qposadr
      self._mocap_target = self._sphere_mocap

  def _rew_dist(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    pass
  
  def _rew_exact(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    pass
  
  def _rew_push(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    pass


import os
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

env = PandaPush(geometry="capsule")
state = env.reset(jax.random.PRNGKey(210))
traj = []
traj.append(state)

instructions = [
   jp.array([50, 50, 0, -100, 0, 0, 0]),
   jp.array([50, 50, 0, -100, 0, 0, 0]),
   jp.array([50, 50, 0, -100, 0, 0, 0]),
   jp.array([50, 50, 0, -100, 0, 0, 0]),
   jp.array([50, 50, 0, 0, 0, 10, 0]),
   jp.array([50, 50, 0, 0, 0, 10, 0]),
   jp.array([100, 0, 0, 0, 0, 10, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
   jp.array([100, 0, 0, 0, 0, 0, 0]),
]

for ins in instructions:
  state = env.step(state, ins)
  traj.append(state)

images = env.render(traj)
import matplotlib.pyplot as plt

for im in images:
    plt.imshow(im)
    plt.show()