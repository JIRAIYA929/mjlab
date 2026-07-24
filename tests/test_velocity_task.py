"""Tests specific to velocity tasks."""

import pytest

from mjlab.asset_zoo.robots import (
  G1_23DOF_ACTION_SCALE,
  G1_ACTION_SCALE,
  GO1_ACTION_SCALE,
  OWN_ACTION_SCALE,
  P1_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.own import own_constants
from mjlab.asset_zoo.robots.p1 import p1_constants
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def g1_29dof_velocity_task_ids(g1_velocity_task_ids: list[str]) -> list[str]:
  """Get all 29-DoF G1 velocity task IDs."""
  return [t for t in g1_velocity_task_ids if "23Dof" not in t]


@pytest.fixture(scope="module")
def g1_23dof_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all 23-DoF G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1-23Dof" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def own_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Own velocity task IDs."""
  return [t for t in velocity_task_ids if t.endswith("-Own")]


@pytest.fixture(scope="module")
def p1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all P1 velocity task IDs."""
  return [t for t in velocity_task_ids if t.endswith("-P1")]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_own_velocity_has_required_sensors(
  own_velocity_task_ids: list[str],
) -> None:
  """Own velocity tasks should define the required sensor contract."""
  for task_id in own_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.sensors is not None
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names
    assert "self_collision" in sensor_names


def test_p1_velocity_has_required_sensors(
  p1_velocity_task_ids: list[str],
) -> None:
  """P1 tasks should define feet, self, and body-ground contacts."""
  assert p1_velocity_task_ids
  for task_id in p1_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.sensors is not None
    sensor_names = {sensor.name for sensor in cfg.scene.sensors}
    assert {
      "feet_ground_contact",
      "self_collision",
      "body_ground_contact",
    } <= sensor_names


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-G1-23Dof",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Velocity-Rough-Own",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-G1-23Dof",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Velocity-Rough-Own",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_g1_velocity_has_correct_action_scale(
  g1_29dof_velocity_task_ids: list[str],
) -> None:
  """29-DoF G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_29dof_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )


def test_g1_23dof_velocity_has_correct_action_scale(
  g1_23dof_velocity_task_ids: list[str],
) -> None:
  """23-DoF G1 velocity tasks should use G1_23DOF_ACTION_SCALE."""
  for task_id in g1_23dof_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.scale == G1_23DOF_ACTION_SCALE


def test_own_velocity_has_correct_action_scale(
  own_velocity_task_ids: list[str],
) -> None:
  """Own velocity tasks should use their isolated action scale."""
  for task_id in own_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.scale == OWN_ACTION_SCALE


def test_p1_velocity_uses_p1_asset_and_action_scale(
  p1_velocity_task_ids: list[str],
) -> None:
  """P1 tasks should use the P1 model and motor-derived action scale."""
  assert p1_velocity_task_ids
  for task_id in p1_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.entities["robot"].spec_fn is p1_constants.get_spec
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.scale == P1_ACTION_SCALE


def test_p1_velocity_has_stability_first_contract(
  p1_velocity_task_ids: list[str],
) -> None:
  """P1 should use conservative commands and explicit fall detection."""
  assert p1_velocity_task_ids
  for task_id in p1_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    assert twist_cmd.ranges.lin_vel_x == (-0.25, 0.5)
    assert twist_cmd.ranges.lin_vel_y == (-0.1, 0.1)
    assert twist_cmd.ranges.ang_vel_z == (-0.25, 0.25)
    assert twist_cmd.rel_standing_envs == 0.2
    assert {"fell_over", "base_too_low", "illegal_body_contact"} <= set(
      cfg.terminations
    )
    assert cfg.rewards["body_orientation_l2"].weight == -2.0
    assert cfg.rewards["foot_slip"].weight == -0.2
    assert cfg.rewards["foot_gait"].params["period"] == 0.7


def test_own_velocity_uses_own_asset(
  own_velocity_task_ids: list[str],
) -> None:
  """Own tasks should use the Own MJCF."""
  for task_id in own_velocity_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.entities["robot"].spec_fn is own_constants.get_spec


def test_own_velocity_has_stability_reward_weights(
  own_velocity_task_ids: list[str],
) -> None:
  """Own velocity tasks should start from the copied stability baseline."""
  for task_id in own_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.rewards["track_linear_velocity"].weight == 1.0
    assert cfg.rewards["track_angular_velocity"].weight == 1.0
    assert cfg.rewards["track_angular_velocity"].params["xy_weight"] == 0.05
    assert "upright" not in cfg.rewards
    assert cfg.rewards["body_orientation_l2"].weight == -1.0
    assert cfg.rewards["body_orientation_l2"].func is mdp.body_orientation_l2
    assert cfg.rewards["pose"].params["walking_threshold"] == 0.1
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    assert cfg.rewards["angular_momentum"].weight == -0.02
    assert cfg.rewards["joint_acc_l2"].weight == -2.5e-7
    assert cfg.rewards["joint_pos_limits"].weight == -10.0


def test_g1_23dof_velocity_has_stability_reward_weights(
  g1_23dof_velocity_task_ids: list[str],
) -> None:
  """23-DoF G1 velocity tasks should use the aligned stability rewards."""
  for task_id in g1_23dof_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.rewards["track_linear_velocity"].weight == 1.0
    assert cfg.rewards["track_angular_velocity"].weight == 1.0
    assert cfg.rewards["track_angular_velocity"].params["xy_weight"] == 0.05
    assert "upright" not in cfg.rewards
    assert cfg.rewards["body_orientation_l2"].weight == -1.0
    assert cfg.rewards["body_orientation_l2"].func is mdp.body_orientation_l2
    assert cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names == (
      "torso_link",
    )
    assert cfg.rewards["pose"].params["walking_threshold"] == 0.1
    assert cfg.rewards["body_ang_vel"].weight == -0.05
    assert cfg.rewards["angular_momentum"].weight == -0.02
    assert cfg.rewards["joint_acc_l2"].weight == -2.5e-7
    assert cfg.rewards["joint_acc_l2"].func is mdp.joint_acc_l2
    assert "dof_pos_limits" not in cfg.rewards
    assert cfg.rewards["joint_pos_limits"].weight == -10.0
