"""Tests for the P1 asset configuration."""

import re

import mujoco
import pytest

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.p1 import p1_constants
from mjlab.entity import Entity


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
  ("actual", "expected"),
  (
    (p1_constants.ARMATURE_X8_P20_120, 0.057694582),
    (p1_constants.ARMATURE_X6_P20_60, 0.025385616),
    (p1_constants.ARMATURE_X4_P36_36, 0.03888),
  ),
)
def test_p1_reflected_inertias(actual: float, expected: float) -> None:
  assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
  (
    "actuator_cfg",
    "stiffness",
    "damping",
    "armature",
    "effort_limit",
    "matched_count",
  ),
  (
    (
      p1_constants.P1_ACTUATOR_HIP_PITCH,
      p1_constants.STIFFNESS_HIP_PITCH_KNEE,
      p1_constants.DAMPING_HIP_PITCH_KNEE,
      p1_constants.ARMATURE_X8_P20_120,
      p1_constants.PEAK_TORQUE_X8_P20_120,
      2,
    ),
    (
      p1_constants.P1_ACTUATOR_KNEE,
      p1_constants.STIFFNESS_HIP_PITCH_KNEE,
      p1_constants.DAMPING_HIP_PITCH_KNEE,
      p1_constants.ARMATURE_X6_P20_60,
      p1_constants.PEAK_TORQUE_X6_P20_60,
      2,
    ),
    (
      p1_constants.P1_ACTUATOR_HIP_ROLL,
      p1_constants.STIFFNESS_HIP_ROLL_YAW,
      p1_constants.DAMPING_HIP_ROLL_YAW,
      p1_constants.ARMATURE_X8_P20_120,
      p1_constants.PEAK_TORQUE_X8_P20_120,
      2,
    ),
    (
      p1_constants.P1_ACTUATOR_HIP_YAW,
      p1_constants.STIFFNESS_HIP_ROLL_YAW,
      p1_constants.DAMPING_HIP_ROLL_YAW,
      p1_constants.ARMATURE_X6_P20_60,
      p1_constants.PEAK_TORQUE_X6_P20_60,
      2,
    ),
    (
      p1_constants.P1_ACTUATOR_ANKLE,
      p1_constants.STIFFNESS_ANKLE_JOINT,
      p1_constants.DAMPING_ANKLE_JOINT,
      p1_constants.ARMATURE_X4_P36_36 * p1_constants.NUM_ANKLE_MOTORS_PER_LEG,
      p1_constants.PEAK_TORQUE_X4_P36_36 * p1_constants.NUM_ANKLE_MOTORS_PER_LEG,
      4,
    ),
  ),
)
def test_p1_actuator_parameters(
  p1_model: mujoco.MjModel,
  actuator_cfg: BuiltinPositionActuatorCfg,
  stiffness: float,
  damping: float,
  armature: float,
  effort_limit: float,
  matched_count: int,
) -> None:
  actual_matched_count = 0
  assert actuator_cfg.effort_limit == effort_limit
  for actuator_id in range(p1_model.nu):
    actuator = p1_model.actuator(actuator_id)
    if not any(
      re.fullmatch(pattern, actuator.name) for pattern in actuator_cfg.target_names_expr
    ):
      continue
    actual_matched_count += 1
    joint = p1_model.joint(actuator.name)
    assert actuator.gainprm[0] == pytest.approx(stiffness)
    assert actuator.biasprm[1] == pytest.approx(-stiffness)
    assert actuator.biasprm[2] == pytest.approx(-damping)
    assert actuator.forcerange[0] == pytest.approx(-effort_limit)
    assert actuator.forcerange[1] == pytest.approx(effort_limit)
    assert p1_model.dof_armature[joint.dofadr[0]] == pytest.approx(armature)
  assert actual_matched_count == matched_count


def test_p1_action_scales() -> None:
  assert p1_constants.P1_ACTION_SCALE == {
    r"hip_pitch_[lr]_joint": 0.25,
    r"knee_pitch_[lr]_joint": 0.25,
    r"hip_roll_[lr]_joint": 0.1,
    r"hip_yaw_[lr]_joint": 0.1,
    r"ankle_pitch_[lr]_joint": 0.1,
    r"ankle_roll_[lr]_joint": 0.1,
  }


def test_p1_parallel_ankle_actuator_parameters() -> None:
  assert p1_constants.NUM_ANKLE_MOTORS_PER_LEG == 2
  assert p1_constants.P1_ACTUATOR_ANKLE.stiffness == pytest.approx(200.0)
  assert p1_constants.P1_ACTUATOR_ANKLE.damping == pytest.approx(12.0)
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


def test_p1_hip_assembly_matches_revised_urdf(
  p1_xml_model: mujoco.MjModel,
) -> None:
  expected_poses = {
    "hip_roll_l_link": (
      (0.03002, 0.09, -0.0648),
      (0.06875776, 0.02075955, 0.00008271),
    ),
    "hip_pitch_l_link": (
      (0.076, 0.035, 0.0),
      (0.0, -0.00443165, -0.03247590),
    ),
    "hip_yaw_l_link": (
      (0.0, -0.024, -0.063),
      (-0.00004624, 0.00328905, -0.08825309),
    ),
    "hip_roll_r_link": (
      (0.0302, -0.09, -0.0648),
      (0.0603906, -0.02075955, 0.00008270),
    ),
    "hip_pitch_r_link": (
      (0.076, -0.035, 0.0),
      (0.0, 0.00443165, -0.03247590),
    ),
    "hip_yaw_r_link": (
      (0.0, 0.024, -0.063),
      (0.0, -0.00315274, -0.08823981),
    ),
  }

  for body_name, (expected_pos, expected_ipos) in expected_poses.items():
    body = p1_xml_model.body(body_name)
    assert body.pos == pytest.approx(expected_pos)
    assert body.ipos == pytest.approx(expected_ipos)


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
