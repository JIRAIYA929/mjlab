"""Tests for the P1 asset configuration."""

import re

import mujoco
import pytest

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.p1 import p1_constants
from mjlab.entity import Entity
from mjlab.utils.actuator import ElectricActuator


@pytest.fixture(scope="module")
def p1_entity() -> Entity:
  return Entity(p1_constants.get_p1_robot_cfg())


@pytest.fixture(scope="module")
def p1_model(p1_entity: Entity) -> mujoco.MjModel:
  return p1_entity.spec.compile()


@pytest.fixture(scope="module")
def p1_xml_model() -> mujoco.MjModel:
  return p1_constants.get_spec().compile()


def test_p1_entity_creation(p1_entity: Entity) -> None:
  assert p1_entity.num_actuators == 12
  assert p1_entity.num_joints == 12
  assert len(set(p1_entity.actuator_names)) == p1_entity.num_actuators
  assert set(p1_entity.actuator_names) == set(p1_entity.joint_names)
  assert p1_entity.is_actuated
  assert not p1_entity.is_fixed_base


def test_p1_initial_pose_is_inside_joint_limits(
  p1_entity: Entity,
  p1_model: mujoco.MjModel,
) -> None:
  key = p1_model.key("init_state")
  for joint_name in p1_entity.joint_names:
    joint = p1_model.joint(joint_name)
    qpos = key.qpos[joint.qposadr[0]]
    assert joint.range[0] <= qpos <= joint.range[1], (
      f"{joint_name} initial position {qpos} is outside {joint.range}"
    )


@pytest.mark.parametrize(
  ("actuator", "armature", "velocity_limit", "effort_limit"),
  (
    (p1_constants.ACTUATOR_X8_P20_120, 0.057694582, 16.545721, 120.0),
    (p1_constants.ACTUATOR_X6_P20_60, 0.025385616, 18.430677, 60.0),
    (p1_constants.ACTUATOR_X4_P36_36, 0.03888, 11.623893, 34.0),
  ),
)
def test_p1_motor_parameters(
  actuator: ElectricActuator,
  armature: float,
  velocity_limit: float,
  effort_limit: float,
) -> None:
  assert actuator.reflected_inertia == pytest.approx(armature)
  assert actuator.velocity_limit == pytest.approx(velocity_limit)
  assert actuator.effort_limit == pytest.approx(effort_limit)


@pytest.mark.parametrize(
  ("actual", "expected"),
  (
    (
      (
        p1_constants.REDUCTION_RATIO_X8_P20_120,
        p1_constants.MASS_X8_P20_120,
        p1_constants.RATED_TORQUE_X8_P20_120,
        p1_constants.RATED_SPEED_X8_P20_120,
      ),
      (19.612, 1.4, 43.0, 13.299409),
    ),
    (
      (
        p1_constants.REDUCTION_RATIO_X6_P20_60,
        p1_constants.MASS_X6_P20_60,
        p1_constants.RATED_TORQUE_X6_P20_60,
        p1_constants.RATED_SPEED_X6_P20_60,
      ),
      (19.612, 0.82, 20.0, 16.022123),
    ),
    (
      (
        p1_constants.REDUCTION_RATIO_X4_P36_36,
        p1_constants.MASS_X4_P36_36,
        p1_constants.RATED_TORQUE_X4_P36_36,
        p1_constants.RATED_SPEED_X4_P36_36,
      ),
      (36.0, 0.36, 10.5, 8.69174),
    ),
  ),
)
def test_p1_datasheet_metadata(
  actual: tuple[float, float, float, float],
  expected: tuple[float, float, float, float],
) -> None:
  assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
  ("actuator_cfg", "stiffness", "damping", "armature"),
  (
    (
      p1_constants.P1_ACTUATOR_X8_P20_120,
      p1_constants.STIFFNESS_X8_P20_120,
      p1_constants.DAMPING_X8_P20_120,
      p1_constants.ARMATURE_X8_P20_120,
    ),
    (
      p1_constants.P1_ACTUATOR_X6_P20_60,
      p1_constants.STIFFNESS_X6_P20_60,
      p1_constants.DAMPING_X6_P20_60,
      p1_constants.ARMATURE_X6_P20_60,
    ),
    (
      p1_constants.P1_ACTUATOR_ANKLE,
      p1_constants.STIFFNESS_X4_P36_36 * p1_constants.NUM_ANKLE_MOTORS_PER_LEG,
      p1_constants.DAMPING_X4_P36_36 * p1_constants.NUM_ANKLE_MOTORS_PER_LEG,
      p1_constants.ARMATURE_X4_P36_36 * p1_constants.NUM_ANKLE_MOTORS_PER_LEG,
    ),
  ),
)
def test_p1_actuator_parameters(
  p1_model: mujoco.MjModel,
  actuator_cfg: BuiltinPositionActuatorCfg,
  stiffness: float,
  damping: float,
  armature: float,
) -> None:
  matched_count = 0
  effort_limit = actuator_cfg.effort_limit
  assert effort_limit is not None
  for actuator_id in range(p1_model.nu):
    actuator = p1_model.actuator(actuator_id)
    if not any(
      re.fullmatch(pattern, actuator.name) for pattern in actuator_cfg.target_names_expr
    ):
      continue
    matched_count += 1
    joint = p1_model.joint(actuator.name)
    assert actuator.gainprm[0] == pytest.approx(stiffness)
    assert actuator.biasprm[1] == pytest.approx(-stiffness)
    assert actuator.biasprm[2] == pytest.approx(-damping)
    assert actuator.forcerange[0] == pytest.approx(-effort_limit)
    assert actuator.forcerange[1] == pytest.approx(effort_limit)
    assert p1_model.dof_armature[joint.dofadr[0]] == pytest.approx(armature)
  assert matched_count == 4


def test_p1_parallel_ankle_actuator_parameters() -> None:
  assert p1_constants.NUM_ANKLE_MOTORS_PER_LEG == 2
  assert p1_constants.P1_ACTUATOR_ANKLE.effort_limit == pytest.approx(68.0)
  assert p1_constants.P1_ACTUATOR_ANKLE.armature == pytest.approx(0.07776)


@pytest.mark.parametrize(
  ("joint_name", "expected_range"),
  (
    ("hip_roll_l_joint", (-0.6, 0.6)),
    ("hip_pitch_l_joint", (-1.0, 0.5)),
    ("hip_yaw_l_joint", (-0.5, 0.5)),
    ("knee_pitch_l_joint", (0.0, 1.8)),
    ("ankle_pitch_l_joint", (-0.6, 0.4)),
    ("ankle_roll_l_joint", (-0.2, 0.2)),
    ("hip_roll_r_joint", (-0.6, 0.6)),
    ("hip_pitch_r_joint", (-1.0, 0.5)),
    ("hip_yaw_r_joint", (-0.5, 0.5)),
    ("knee_pitch_r_joint", (0.0, 1.8)),
    ("ankle_pitch_r_joint", (-0.6, 0.4)),
    ("ankle_roll_r_joint", (-0.2, 0.2)),
  ),
)
def test_p1_joint_limits_match_urdf(
  p1_model: mujoco.MjModel,
  joint_name: str,
  expected_range: tuple[float, float],
) -> None:
  assert p1_model.joint(joint_name).range == pytest.approx(expected_range)


def test_p1_actuators_are_force_limited(p1_model: mujoco.MjModel) -> None:
  for actuator_id in range(p1_model.nu):
    assert p1_model.actuator_ctrllimited[actuator_id] == 0
    assert p1_model.actuator_forcelimited[actuator_id] == 1


def test_p1_foot_collision_geoms(p1_model: mujoco.MjModel) -> None:
  foot_pattern = re.compile(p1_constants.FOOT_COLLISION_PATTERN)
  foot_geoms = [
    p1_model.geom(i)
    for i in range(p1_model.ngeom)
    if foot_pattern.match(p1_model.geom(i).name)
  ]
  assert len(foot_geoms) == 8
  for geom in foot_geoms:
    assert geom.condim == 3
    assert geom.priority == 1
    assert geom.friction[0] == 0.6


def test_p1_visual_meshes_are_loaded(p1_model: mujoco.MjModel) -> None:
  assert p1_model.nmesh == 13


def test_p1_xml_geometry_and_appearance(p1_xml_model: mujoco.MjModel) -> None:
  assert p1_xml_model.body("pelvis_link").pos[2] == pytest.approx(0.65454)

  data = mujoco.MjData(p1_xml_model)
  mujoco.mj_forward(p1_xml_model, data)
  for side in ("left", "right"):
    site_id = p1_xml_model.site(f"{side}_foot").id
    assert data.site_xpos[site_id, 2] == pytest.approx(0.0, abs=1e-8)
    for foot_id in range(1, 5):
      geom = p1_xml_model.geom(f"{side}_foot{foot_id}_collision")
      assert data.geom_xpos[geom.id, 2] - geom.size[0] == pytest.approx(0.0, abs=1e-8)

  for side in ("l", "r"):
    for joint in ("hip_pitch", "ankle_pitch"):
      geom = p1_xml_model.geom(f"{joint}_{side}_collision")
      assert geom.size[0] == pytest.approx(0.005)

  white_material_id = p1_xml_model.mat("white").id
  assert p1_xml_model.mat("white").rgba == pytest.approx((1.0, 1.0, 1.0, 1.0))
  for geom_id in range(p1_xml_model.ngeom):
    if p1_xml_model.geom_group[geom_id] == 2:
      assert p1_xml_model.geom_matid[geom_id] == white_material_id
    elif p1_xml_model.geom_group[geom_id] == 3:
      assert p1_xml_model.geom(geom_id).rgba == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_p1_xml_imu_and_contact_exclusions(p1_xml_model: mujoco.MjModel) -> None:
  imu_site = p1_xml_model.site("imu_in_pelvis")
  assert imu_site.type[0] == mujoco.mjtGeom.mjGEOM_BOX
  assert imu_site.pos == pytest.approx((0.0, 0.0, 0.0))
  assert imu_site.size == pytest.approx((0.01, 0.01, 0.005))

  assert [p1_xml_model.sensor(i).name for i in range(p1_xml_model.nsensor)] == [
    "imu_ang_vel",
    "imu_lin_vel",
    "imu_lin_acc",
    "root_angmom",
  ]
  assert p1_xml_model.nexclude == 2
