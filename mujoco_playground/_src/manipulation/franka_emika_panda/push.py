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

#Initial position of object
INIT_POS = [0.5, 0, 0]
#Endeffector info
ENDEFFECTOR_HEIGHT = 0.07 #m = 70 mm
ENDEFFECTOR_HALFLENGTH = 0.03
ENDEFFECTOR_HALFWIDTH = 0.005

def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02, #Used in MjxEnv init()
      sim_dt=0.005, #Used in MjxEnv init()
      episode_length=150, #Not used here
      action_repeat=1, #Not used here
      action_scale=0.04, #Used in step()
      reward_config=config_dict.create( #Used in step()
          scales=config_dict.create(
              # r_dist
              r_dist = 1.0,
              # r_exact
              r_exact = 1.0,
              # r_push
              r_push = 1.0,
              # r_vel
              r_vel = -1.0,
              #r_smooth
              r_smooth = -1.0,
              #r_neutral
              r_neutral = -1.0,
              #r_limit
              r_limit = -1.0,
              #r_col
              r_col = -1.0
          )
      ),
      r_exact_epsilon = 0.001, #Used in _r_exact()
      domain_parameters=config_dict.create(
         min_size=0.005, #m (MuJoCo uses half sizes)
         max_size=0.05, #m
         min_density=300, #kg/m^3 (Use density instead of mass directly, as it will lead to small objects with very high mass -> get stuck halfway in floor)
         max_density=23_000, #kg/m^3 (higher density than Osmium)
         min_slide=0.5, #slide friction
         max_slide=2.0, 
         min_roll=0.01, #roll friction
         max_roll=0.1,
         min_spin=0.001, #spin friction
         max_spin=0.01
      ),
      impl='jax',
      nconmax=24 * 8192,
      njmax=128,
  )
  return config


class PandaPush(panda.PandaBase):
  """Bring a box to a target. This environment lets you choose whether or not you want to randomize the object geometriy (available: 'box', 'capsule', 'spphere'),
  whether or not you want domain_randomization (on each reset randomize: mass, friction, size)

  Done Criterion are:
   - Object Position is >1m in any coordinate
   - Object z-Position <0
   - Any data.qpos entry is nan
   - Any data.qvel entry is nan
   - Object X and Y position are within a distance of 1mm from the target position.
   """
  geometries = ["box", "capsule", "sphere"] #Order should not be changed, as it would break _get_mask, and therefore _get_obs
  epsilon = 0.001

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      geometry_randomization: bool = True,
      domain_randomization: bool = True,
      geometry: str = "box",
  ):
    """If domain_randomization = True the environment will randomly draw from the possible geometries on each reset; if false
    the object geometry will always be the shape supplied in the geometry attribute. Shapes can also be changed manually by calling _set_obj_geom()
    before calling reset()."""
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
    self._domain_randomization = domain_randomization
    self._geometry_randomization = geometry_randomization
    self._geometry_name = geometry

    # Contact sensor IDs.
    self._floor_endeffector_found_sensor = [
      self._mj_model.sensor(f"endeffector_{name}_floor_found").id
      for name in ["head", "stick"]
    ]
    # Read out the range of the joints of robot
    self.q_min = self.mj_model.jnt_range[:7, 0]
    self.q_max = self.mj_model.jnt_range[:7, 1]
    # Init last action
    self.last_action = jp.zeros(7)

  def _move_other_geometries_away(self, geom: str):
    if geom=="box":
        self._mj_model.body("capsule").pos = [100, 100, 0]
        self._mj_model.body("sphere").pos = [101, 100, 0]
        self._mj_model.body("capsule_target").pos = [102, 100, 0]
        self._mj_model.body("sphere_target").pos = [103, 100, 0]

    elif geom=="capsule":
        self._mj_model.body("box").pos = [100, 100, 0]
        self._mj_model.body("sphere").pos = [101, 100, 0]
        self._mj_model.body("box_target").pos = [102, 100, 0]
        self._mj_model.body("sphere_target").pos = [103, 100, 0]

    elif geom=="sphere":
        self._mj_model.body("box").pos = [100, 100, 0]
        self._mj_model.body("capsule").pos = [101, 100, 0]
        self._mj_model.body("box_target").pos = [102, 100, 0]
  
  def _compute_mass_from_density(self, geom: str, size: jax.Array, density: float) -> float:
    if geom=="box":
        V = 8 * size[0] * size[1] * size[2] #multiply by 8 because size contains half-sizes
    elif geom=="capsule":
        V = np.pi * size[0]*size[0] * (2*size[1]) + 4/3 * np.pi * size[0]*size[0]*size[0] #V = Volume of cylinder + Volume of sphere
    elif geom=="sphere":
        V = np.pi * 4/3 * size[0]*size[0]*size[0]
    return density*V  
        

  def _randomize_domain(self, geom: str, rng: jax.random.PRNGKey):
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if geom=="box":
        #Randomize Size (because its geometry based)
        size = jax.random.uniform(rng, (3,), minval=self._config.domain_parameters.min_size, maxval=self._config.domain_parameters.max_size)
        height_offset = size[-1]
        #Randomly rotate around z-axis

    elif geom=="capsule":
        #Randomize Size (because its geometry based)
        size = jax.random.uniform(rng, (2,), minval=self._config.domain_parameters.min_size, maxval=self._config.domain_parameters.max_size)
        #Flip capsule on the side
        quat = jp.array([jp.sqrt(0.5), jp.sqrt(0.5), 0.0, 0.0], dtype=float)
        height_offset = size[0] #because capsule is defined by giving radius followed by cylinder halfheight

    elif geom=="sphere":
        #Randomize Size (because its geometry based)
        size = jax.random.uniform(rng, (1,), minval=self._config.domain_parameters.min_size, maxval=self._config.domain_parameters.max_size)
        height_offset = size[0]

    #Set size
    self._mj_model.geom(geom).size = size
    self._mj_model.geom(f"{geom}_target").size = size
    #Randomize Mass
    density = jax.random.uniform(rng, (1,), minval=self._config.domain_parameters.min_density, maxval=self._config.domain_parameters.max_density)
    self._mj_model.body(geom).mass = self._compute_mass_from_density(geom, size, density)
    #Randomize Friction
    self._mj_model.geom(geom).friction = jax.random.uniform(rng, 
                                                            (3,), 
                                                            minval=jp.array([self._config.domain_parameters.min_slide,
                                                                             self._config.domain_parameters.min_roll, 
                                                                             self._config.domain_parameters.min_spin]),
                                                            maxval=jp.array([self._config.domain_parameters.max_slide,
                                                                             self._config.domain_parameters.max_roll, 
                                                                             self._config.domain_parameters.max_spin]))
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

    if self._geometry_randomization:
       geom_idx = jax.random.choice(rng_box, len(self.geometries))
       geom = self.geometries[geom_idx]
       self._set_obj_geom(geom)

    #Move the other geometries out of range to not interfere (just to be safe)
    self._move_other_geometries_away(self._geometry_name)

    if self._domain_randomization:
        height_offset, quat = self._randomize_domain(geom, rng_box)
    else:
       height_offset = self._mj_model.geom(self._geometry_name).size[0] if self._geometry_name!="box" else self._mj_model.geom(self._geometry_name).size[-1]
    #if we changed the mj_model we need to make a new mjx_model as well
    if self._domain_randomization or self._geometry_randomization:
       self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

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

    raw_rewards = self._get_reward(data, state.info, action)
    rewards = {
       k: v * self._config.reward_config.scales[k]
       for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    #Done criterion
    #Object is very close to the mocap x,y coordinates AND velocity of object is not too high
    obj_pos = data.xpos[self._obj_body]
    goal_pos = data.mocap_pos[self._mocap_target][0, :2]#[:2]
    distance = jp.linalg.norm(obj_pos[:2]-goal_pos)
    close_enough = distance < 0.001 #1mm distance from target
    out_of_bounds = jp.any(jp.abs(obj_pos) > 1.0)
    out_of_bounds |= obj_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() | close_enough
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    #Set action to self.last_action
    self.last_action = action

    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any], action: jax.Array) -> Dict[str, Any]:
    #rew_push:
    r_dist = self._r_dist(data, info)
    r_exact = self._r_exact(data, info)
    r_push = self._r_push(data, info)
    #pen_push (scales in config should be negative for these to be treated as penalties):
    r_vel = self._r_vel(data, info)
    r_smooth = self._r_smooth(action, info)
    r_neutral = self._r_neutral(data, info)
    r_limit = self._r_limit(data, info)
    r_col = self._r_col(data, info)

    rewards = {
        "r_dist": r_dist,
        "r_exact": r_exact,
        "r_push": r_push,
        "r_vel": r_vel,
        "r_smooth": r_smooth,
        "r_neutral": r_neutral,
        "r_limit": r_limit,
        "r_col": r_col
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
    #Mask out all unused objects from qpos and qvel
    qpos_mask = self._get_mask(data.qpos.shape)
    qvel_mask = self._get_mask(data.qvel.shape, qvel=True)

    obs = jp.concatenate([
        data.qpos[qpos_mask], #First 7 entries: robot joint pos., last 7 entries: x,y,z coordinates + quaternion of object
        data.qvel[qvel_mask], #First 7 entries are robot joint velocities, last 6 entries linear velocities followed by angular velocities of object
        self.last_action, #Last action performed
        info["target_pos"][:2], #x, y coordinate of target_pos
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
    self._endeffector_body = self._mj_model.body("endeffector").id
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
    self._geometry_name = obj_name

  def _r_dist(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    ee_obj = jp.square(data.xpos[self._endeffector_body] - data.xpos[self._obj_body]).sum()
    return 1 / (1+ee_obj)
  
  def _r_exact(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    norm = jp.linalg.norm(data.xpos[self._endeffector_body]-data.xpos[self._obj_body])
    if norm < self._config.r_exact_epsilon:
       return 1 + 1/(1+100*data.qvel[self._robot_qposadr])/self._config.reward_config.scales["r_exact"]
    else:
       return 0
  
  def _r_push(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    """Distance of the xand y coordinate of object to the target's x and y coordinate"""
    obj_goal = jp.square(data.qpos[self._obj_qposadr:self._obj_qposadr+2] - info["target_pos"][:2]).sum()
    return 1.0 / (1+obj_goal)

  def _r_vel(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    return jp.square(data.qvel[self._robot_qposadr]).sum()
  
  def _r_smooth(self, action: jp.array, info: Dict[str, Any]) -> float:
    norm = jp.linalg.norm(action-self.last_action)
    return norm
  
  def _r_neutral(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    norm = jp.linalg.norm(data.qpos[self._robot_qposadr] - self._init_q[:7])
    return norm
  
  def _r_limit(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    q_dif_min = data.qpos[self._robot_qposadr] - self.q_min
    q_dif_max = data.qpos[self._robot_qposadr] - self.q_max
    q_dif = jp.minimum(q_dif_min, q_dif_max)
    return jp.exp(-30*jp.square(q_dif)).sum()
  
  def _r_col(self, data: mjx.Data, info: Dict[str, Any]) -> float:
    p_ee = data.xpos[self._endeffector_body]
    R_ee = data.xmat[self._endeffector_body].reshape(3,3)
    #Compute the positions of the 4 corners and take the min
    ee_corners = jp.array([
       [-ENDEFFECTOR_HALFLENGTH, -ENDEFFECTOR_HALFWIDTH, -ENDEFFECTOR_HEIGHT],
       [-ENDEFFECTOR_HALFLENGTH, ENDEFFECTOR_HALFWIDTH, -ENDEFFECTOR_HEIGHT],
       [ENDEFFECTOR_HALFLENGTH, -ENDEFFECTOR_HALFWIDTH, -ENDEFFECTOR_HEIGHT],
       [ENDEFFECTOR_HALFLENGTH, ENDEFFECTOR_HALFWIDTH, -ENDEFFECTOR_HEIGHT]
    ])
    corners = p_ee + (R_ee @ ee_corners.T).T
    min_z = corners[:, 2].min()
    if min_z < 0.01:
       return 1.0
    else:
       return 0.0