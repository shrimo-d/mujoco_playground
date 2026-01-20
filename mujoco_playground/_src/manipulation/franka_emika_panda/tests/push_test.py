import pytest
import jax
import jax.numpy as jp
import numpy as np

# >>> passe diesen Import an <<<
from mujoco_playground._src.manipulation.franka_emika_panda.push import PandaPush, default_config


def make_env(
    *,
    geometry_randomization=True,
    domain_randomization=True,
    geometry="box",
):
    cfg = default_config()
    # ggf. cfg overrides hier setzen, wenn du schneller testen willst
    return PandaPush(
        config=cfg,
        geometry_randomization=geometry_randomization,
        domain_randomization=domain_randomization,
        geometry=geometry,
    )


def assert_all_finite(x, name="tensor"):
    assert jp.all(jp.isfinite(x)), f"{name} contains NaN/Inf"


def test_reset_step_contract_shapes_and_finiteness():
    env = make_env(geometry_randomization=True, domain_randomization=True)
    rng = jax.random.PRNGKey(0)

    state = env.reset(rng)
    # obs/reward/done basic sanity
    assert state.obs.ndim == 1
    assert_all_finite(state.obs, "obs(reset)")
    assert_all_finite(state.data.qpos, "qpos(reset)")
    assert_all_finite(state.data.qvel, "qvel(reset)")

    # step once with zero action
    action = jp.zeros(7)
    state2 = env.step(state, action)

    assert state2.obs.ndim == 1
    assert state2.obs.shape == state.obs.shape, "obs shape must remain constant"
    assert_all_finite(state2.obs, "obs(step)")
    assert_all_finite(state2.data.qpos, "qpos(step)")
    assert_all_finite(state2.data.qvel, "qvel(step)")

    # reward & done must be finite
    assert_all_finite(jp.asarray(state2.reward), "reward")
    assert state2.done in (0.0, 1.0), "done must be 0.0 or 1.0"


def test_obs_dim_constant_across_geometries():
    # test each geometry fixed, no randomization
    obs_shapes = []
    for geom in ["box", "capsule", "sphere"]:
        env = make_env(geometry_randomization=False, domain_randomization=False, geometry=geom)
        st = env.reset(jax.random.PRNGKey(0))
        obs_shapes.append(st.obs.shape)

    assert len(set(obs_shapes)) == 1, f"obs shape differs across geometries: {obs_shapes}"


def test_many_resets_do_not_crash_with_randomization():
    env = make_env(geometry_randomization=True, domain_randomization=True)
    rng = jax.random.PRNGKey(42)

    # This catches shape issues in quaternion randomization etc.
    for i in range(200):
        rng, sub = jax.random.split(rng)
        st = env.reset(sub)
        assert_all_finite(st.obs, f"obs(reset {i})")
        assert_all_finite(st.data.qpos, f"qpos(reset {i})")
        assert_all_finite(st.data.qvel, f"qvel(reset {i})")


def test_domain_randomization_true_geometry_randomization_false_does_not_crash():
    # This test will FAIL with your current code because `geom` can be undefined in reset()
    env = make_env(geometry_randomization=False, domain_randomization=True, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))
    assert_all_finite(st.obs, "obs")


def test_step_action_clipping_stability_large_actions():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))

    # big action should not produce NaNs; ctrl is clipped internally
    big = jp.ones(7) * 1e6
    for _ in range(50):
        st = env.step(st, big)
        assert_all_finite(st.obs, "obs")
        assert_all_finite(st.data.qpos, "qpos")
        assert_all_finite(st.data.qvel, "qvel")
        assert_all_finite(jp.asarray(st.reward), "reward")
        assert st.done in (0.0, 1.0)


def test_rewards_are_scalars_not_vectors():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))

    # call internal reward function once
    raw = env._get_reward(st.data, st.info, jp.zeros(7))  # pylint: disable=protected-access
    # each reward component should be scalar shape ()
    for k, v in raw.items():
        v = jp.asarray(v)
        assert v.shape == (), f"reward component {k} is not scalar, got shape={v.shape}"


def test_done_triggers_on_out_of_bounds_by_state_hack():
    # We "hack" state.data to force object out of bounds.
    # This tests termination logic without needing a controller.
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))

    obj_qposadr = env._obj_qposadr  # pylint: disable=protected-access

    # Put object far away in x to trigger out_of_bounds
    qpos = st.data.qpos.at[obj_qposadr + 0].set(2.0)  # x > 1.0
    data2 = st.data.replace(qpos=qpos)
    st2 = st.replace(data=data2)

    st3 = env.step(st2, jp.zeros(7))
    assert st3.done == 1.0, "done should trigger when object is out of bounds"
    assert st3.metrics["out_of_bounds"] == 1.0


def test_jit_compilation_step_does_not_fail():
    # MJX is often used with jit; this test catches hidden shape issues in JAX paths.
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    rng = jax.random.PRNGKey(0)
    st = env.reset(rng)
    action = jp.zeros(7)

    step_jit = jax.jit(env.step)

    st2 = step_jit(st, action)
    assert_all_finite(st2.obs, "obs(jit)")
    assert_all_finite(jp.asarray(st2.reward), "reward(jit)")

def test_reset_deterministic_for_same_seed():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    rng = jax.random.PRNGKey(123)

    s1 = env.reset(rng)
    s2 = env.reset(rng)

    assert jp.allclose(s1.obs, s2.obs), "reset should be deterministic for same seed"
    assert jp.allclose(s1.data.qpos, s2.data.qpos)
    assert jp.allclose(s1.data.qvel, s2.data.qvel)

def test_fuzz_rollout_no_nans_and_terminates():
    env = make_env(geometry_randomization=True, domain_randomization=True)
    rng = jax.random.PRNGKey(0)
    st = env.reset(rng)

    rng = jax.random.PRNGKey(1)
    max_steps = 300  # > episode_length, weil du episode_length aktuell nicht nutzt
    for t in range(max_steps):
        rng, sub = jax.random.split(rng)
        a = jax.random.uniform(sub, (7,), minval=-1.0, maxval=1.0)
        st = env.step(st, a)

        assert_all_finite(st.obs, f"obs(t={t})")
        assert_all_finite(st.data.qpos, f"qpos(t={t})")
        assert_all_finite(st.data.qvel, f"qvel(t={t})")
        assert_all_finite(jp.asarray(st.reward), f"reward(t={t})")

        if float(st.done) == 1.0:
            break

    assert float(st.done) == 1.0, "episode should terminate within max_steps"

def test_domain_randomization_ranges():
    env = make_env(geometry_randomization=True, domain_randomization=True)
    cfg = default_config()
    rng = jax.random.PRNGKey(0)

    for _ in range(50):
        rng, sub = jax.random.split(rng)
        _ = env.reset(sub)

        geom = env._geometry_name  # python state ok hier, wir sind nicht in jit
        size = np.array(env._mj_model.geom(geom).size, dtype=float)
        fric = np.array(env._mj_model.geom(geom).friction, dtype=float)
        mass = env._mj_model.body(geom).mass.item()

        assert np.isfinite(size).all()
        assert np.isfinite(fric).all()
        assert np.isfinite(mass)

        assert (size >= cfg.domain_parameters.min_size).all()
        assert (size <= cfg.domain_parameters.max_size).all()

        assert fric[0] >= cfg.domain_parameters.min_slide and fric[0] <= cfg.domain_parameters.max_slide
        assert fric[1] >= cfg.domain_parameters.min_roll  and fric[1] <= cfg.domain_parameters.max_roll
        assert fric[2] >= cfg.domain_parameters.min_spin  and fric[2] <= cfg.domain_parameters.max_spin

        assert mass > 0.0

def test_done_triggers_on_close_enough_by_state_hack():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))

    obj_qposadr = env._obj_qposadr
    # target xy aus info (in deinem code: info["target_pos"])
    tgt_xy = st.info["target_pos"][:2]

    qpos = st.data.qpos
    qpos = qpos.at[obj_qposadr + 0].set(tgt_xy[0])
    qpos = qpos.at[obj_qposadr + 1].set(tgt_xy[1])

    data2 = st.data.replace(qpos=qpos)
    st2 = st.replace(data=data2)

    st3 = env.step(st2, jp.zeros(7))
    assert st3.done == 1.0, "done should trigger when object is within 1mm of target (xy)"

def test_obs_does_not_include_full_qpos_qvel():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))

    # obs should be smaller than full qpos+qvel if you're masking out other objects
    assert st.obs.shape[0] < (st.data.qpos.shape[0] + st.data.qvel.shape[0] + 7 + 2)

def test_reward_clipped_range():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    st = env.reset(jax.random.PRNGKey(0))

    big = jp.ones(7) * 1e6
    for _ in range(50):
        st = env.step(st, big)
        r = float(st.reward)
        assert -1e4 <= r <= 1e4

def test_vmap_reset_smoke():
    env = make_env(geometry_randomization=False, domain_randomization=False, geometry="box")
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    states = jax.vmap(env.reset)(keys)
    assert states.obs.shape[0] == 8
