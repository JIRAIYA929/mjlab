"""Own constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia,
  rpm_to_rad,
)
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

OWN_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "own" / "xmls" / "own.xml"
assert OWN_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(OWN_XML))


##
# Actuator config.
##

# Motor parameters are specified at the integrated joint-module output. Rotor
# inertia is provided on the motor side in kg*mm^2 and reflected through the reducer.
REDUCTION_RATIO_A4310 = 36.0
ROTOR_INERTIA_A4310 = 18.0e-6
ARMATURE_A4310 = reflected_inertia(ROTOR_INERTIA_A4310, REDUCTION_RATIO_A4310)
ACTUATOR_A4310 = ElectricActuator(
  reflected_inertia=ARMATURE_A4310,
  velocity_limit=rpm_to_rad(89.0),
  effort_limit=36.0,
)

REDUCTION_RATIO_A4315 = 36.0
ROTOR_INERTIA_A4315 = 25.0e-6
ARMATURE_A4315 = reflected_inertia(ROTOR_INERTIA_A4315, REDUCTION_RATIO_A4315)
ACTUATOR_A4315 = ElectricActuator(
  reflected_inertia=ARMATURE_A4315,
  velocity_limit=rpm_to_rad(117.0),
  effort_limit=75.0,
)

REDUCTION_RATIO_A8112 = 18.0
ROTOR_INERTIA_A8112 = 98.0e-6
ARMATURE_A8112 = reflected_inertia(ROTOR_INERTIA_A8112, REDUCTION_RATIO_A8112)
ACTUATOR_A8112 = ElectricActuator(
  reflected_inertia=ARMATURE_A8112,
  velocity_limit=rpm_to_rad(157.0),
  effort_limit=90.0,
)

REDUCTION_RATIO_A8116 = 18.0
ROTOR_INERTIA_A8116 = 197.0e-6
ARMATURE_A8116 = reflected_inertia(ROTOR_INERTIA_A8116, REDUCTION_RATIO_A8116)
ACTUATOR_A8116 = ElectricActuator(
  reflected_inertia=ARMATURE_A8116,
  velocity_limit=rpm_to_rad(140.0),
  effort_limit=130.0,
)

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_A4310 = ARMATURE_A4310 * NATURAL_FREQ**2
STIFFNESS_A4315 = ARMATURE_A4315 * NATURAL_FREQ**2
STIFFNESS_A8112 = ARMATURE_A8112 * NATURAL_FREQ**2
STIFFNESS_A8116 = ARMATURE_A8116 * NATURAL_FREQ**2

DAMPING_A4310 = 2.0 * DAMPING_RATIO * ARMATURE_A4310 * NATURAL_FREQ
DAMPING_A4315 = 2.0 * DAMPING_RATIO * ARMATURE_A4315 * NATURAL_FREQ
DAMPING_A8112 = 2.0 * DAMPING_RATIO * ARMATURE_A8112 * NATURAL_FREQ
DAMPING_A8116 = 2.0 * DAMPING_RATIO * ARMATURE_A8116 * NATURAL_FREQ

OWN_ACTUATOR_A4310 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_A4310,
  damping=DAMPING_A4310,
  effort_limit=ACTUATOR_A4310.effort_limit,
  armature=ACTUATOR_A4310.reflected_inertia,
)
OWN_ACTUATOR_A4315 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_A4315,
  damping=DAMPING_A4315,
  effort_limit=ACTUATOR_A4315.effort_limit,
  armature=ACTUATOR_A4315.reflected_inertia,
)
OWN_ACTUATOR_A8112 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_A8112,
  damping=DAMPING_A8112,
  effort_limit=ACTUATOR_A8112.effort_limit,
  armature=ACTUATOR_A8112.reflected_inertia,
)
OWN_ACTUATOR_A8116 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_A8116,
  damping=DAMPING_A8116,
  effort_limit=ACTUATOR_A8116.effort_limit,
  armature=ACTUATOR_A8116.reflected_inertia,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": -0.7008,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": -0.9708,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

OWN_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    OWN_ACTUATOR_A4310,
    OWN_ACTUATOR_A4315,
    OWN_ACTUATOR_A8112,
    OWN_ACTUATOR_A8116,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_own_robot_cfg() -> EntityCfg:
  """Get a fresh Own robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=OWN_ARTICULATION,
  )


OWN_ACTION_SCALE: dict[str, float] = {}
for a in OWN_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    OWN_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_own_robot_cfg())

  viewer.launch(robot.spec.compile())
