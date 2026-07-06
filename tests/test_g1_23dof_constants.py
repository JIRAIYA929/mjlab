"""Tests for the Unitree G1 23-DoF asset configuration."""

import re

import mujoco
import pytest

from mjlab.asset_zoo.robots.unitree_g1 import g1_23dof_constants
from mjlab.entity import Entity


@pytest.fixture(scope="module")
def g1_23dof_entity() -> Entity:
  return Entity(g1_23dof_constants.get_g1_23dof_robot_cfg())


@pytest.fixture(scope="module")
def g1_23dof_model(g1_23dof_entity: Entity) -> mujoco.MjModel:
  return g1_23dof_entity.spec.compile()


def test_g1_23dof_entity_creation(g1_23dof_entity: Entity) -> None:
  assert g1_23dof_entity.num_actuators == 23
  assert g1_23dof_entity.num_joints == 23
  assert g1_23dof_entity.is_actuated
  assert not g1_23dof_entity.is_fixed_base


def test_g1_23dof_actuators_are_force_limited(
  g1_23dof_model: mujoco.MjModel,
) -> None:
  for actuator_id in range(g1_23dof_model.nu):
    assert g1_23dof_model.actuator_ctrllimited[actuator_id] == 0
    assert g1_23dof_model.actuator_forcelimited[actuator_id] == 1


def test_g1_23dof_foot_collision_geoms(g1_23dof_model: mujoco.MjModel) -> None:
  foot_pattern = re.compile(r"^(left|right)_foot[1-7]_collision$")
  foot_geoms = [
    g1_23dof_model.geom(i)
    for i in range(g1_23dof_model.ngeom)
    if foot_pattern.match(g1_23dof_model.geom(i).name)
  ]
  assert len(foot_geoms) == 14
  for geom in foot_geoms:
    assert geom.condim == 3
    assert geom.priority == 1
    assert geom.friction[0] == 0.6
