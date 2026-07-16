"""Tests for the Own robot configuration."""

import re

import mujoco
import pytest

from mjlab.asset_zoo.robots.own import own_constants
from mjlab.entity import Entity
from mjlab.utils.actuator import ElectricActuator


@pytest.fixture(scope="module")
def own_entity() -> Entity:
  return Entity(own_constants.get_own_robot_cfg())


@pytest.fixture(scope="module")
def own_model(own_entity: Entity) -> mujoco.MjModel:
  return own_entity.spec.compile()


def test_own_entity_creation(own_entity: Entity) -> None:
  assert own_entity.num_actuators == 23
  assert own_entity.num_joints == 23
  assert len(set(own_entity.actuator_names)) == own_entity.num_actuators
  assert set(own_entity.actuator_names) == set(own_entity.joint_names)
  assert own_entity.is_actuated
  assert not own_entity.is_fixed_base


def test_own_initial_pose_is_inside_joint_limits(
  own_entity: Entity,
  own_model: mujoco.MjModel,
) -> None:
  key = own_model.key("init_state")
  for joint_name in own_entity.joint_names:
    joint = own_model.joint(joint_name)
    qpos = key.qpos[joint.qposadr[0]]
    assert joint.range[0] <= qpos <= joint.range[1], (
      f"{joint_name} initial position {qpos} is outside {joint.range}"
    )


def test_own_elbow_home_position() -> None:
  own_joint_pos = own_constants.HOME_KEYFRAME.joint_pos
  assert own_joint_pos is not None
  assert own_joint_pos[".*_elbow_joint"] == pytest.approx(-0.7008)


@pytest.mark.parametrize(
  ("actuator", "armature", "velocity_limit", "effort_limit"),
  (
    (own_constants.ACTUATOR_A4310, 0.023328, 9.320058, 36.0),
    (own_constants.ACTUATOR_A4315, 0.0324, 12.252211, 75.0),
    (own_constants.ACTUATOR_A8112, 0.031752, 16.441002, 90.0),
    (own_constants.ACTUATOR_A8116, 0.063828, 14.660766, 130.0),
  ),
)
def test_own_motor_parameters(
  actuator: ElectricActuator,
  armature: float,
  velocity_limit: float,
  effort_limit: float,
) -> None:
  assert actuator.reflected_inertia == pytest.approx(armature)
  assert actuator.velocity_limit == pytest.approx(velocity_limit)
  assert actuator.effort_limit == pytest.approx(effort_limit)


def test_own_actuators_are_force_limited(
  own_model: mujoco.MjModel,
) -> None:
  for actuator_id in range(own_model.nu):
    assert own_model.actuator_forcelimited[actuator_id] == 1


def test_own_foot_collision_geoms(
  own_model: mujoco.MjModel,
) -> None:
  foot_pattern = re.compile(r"^(left|right)_foot[1-7]_collision$")
  foot_geoms = [
    own_model.geom(i)
    for i in range(own_model.ngeom)
    if foot_pattern.match(own_model.geom(i).name)
  ]
  assert len(foot_geoms) == 14
  for geom in foot_geoms:
    assert geom.condim == 3
    assert geom.priority == 1
    assert geom.friction[0] == 0.6
