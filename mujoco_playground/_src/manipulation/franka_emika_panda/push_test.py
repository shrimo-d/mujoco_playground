from mujoco_playground._src.manipulation.franka_emika_panda.push import PandaPush
import jax.numpy as jp
import jax
import os
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

env = PandaPush(geometry="capsule", geometry_randomization=False, domain_randomization=False)
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
  print(state.data.xpos[env._obj_body], f"Size: {env._mj_model.geom('capsule').size}")
  traj.append(state)

images = env.render(traj)
import matplotlib.pyplot as plt

for im in images:
    plt.imshow(im)
    plt.show()