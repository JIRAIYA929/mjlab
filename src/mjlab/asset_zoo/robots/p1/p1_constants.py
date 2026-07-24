"""P1 biped robot constants."""

import math
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia, rpm_to_rad
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

P1_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "p1" / "xmls" / "p1.xml"
assert P1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load a fresh P1 MuJoCo specification."""
  return mujoco.MjSpec.from_file(str(P1_XML))


##
# Actuator config.
##

# The datasheet inertias are treated as motor-side values in kg*cm^2 and
# reflected to the joint output using the datasheet's actual reducer ratios.
# Rated values describe continuous operation; the actuator limits use peak
# torque and no-load speed. Module masses are retained as datasheet metadata;
# the URDF-derived body inertials already contain the robot's physical mass.
REDUCTION_RATIO_X8_P20_120 = 19.612
ROTOR_INERTIA_X8_P20_120 = 1.5e-4
ARMATURE_X8_P20_120 = reflected_inertia(
  ROTOR_INERTIA_X8_P20_120, REDUCTION_RATIO_X8_P20_120
)
MASS_X8_P20_120 = 1.4
RATED_TORQUE_X8_P20_120 = 43.0
RATED_SPEED_X8_P20_120 = rpm_to_rad(127.0)
PEAK_TORQUE_X8_P20_120 = 120.0
NO_LOAD_SPEED_X8_P20_120 = rpm_to_rad(158.0)
ACTUATOR_X8_P20_120 = ElectricActuator(
  reflected_inertia=ARMATURE_X8_P20_120,
  velocity_limit=NO_LOAD_SPEED_X8_P20_120,
  effort_limit=PEAK_TORQUE_X8_P20_120,
)

REDUCTION_RATIO_X6_P20_60 = 19.612
ROTOR_INERTIA_X6_P20_60 = 0.66e-4
ARMATURE_X6_P20_60 = reflected_inertia(
  ROTOR_INERTIA_X6_P20_60, REDUCTION_RATIO_X6_P20_60
)
MASS_X6_P20_60 = 0.82
RATED_TORQUE_X6_P20_60 = 20.0
RATED_SPEED_X6_P20_60 = rpm_to_rad(153.0)
PEAK_TORQUE_X6_P20_60 = 60.0
NO_LOAD_SPEED_X6_P20_60 = rpm_to_rad(176.0)
ACTUATOR_X6_P20_60 = ElectricActuator(
  reflected_inertia=ARMATURE_X6_P20_60,
  velocity_limit=NO_LOAD_SPEED_X6_P20_60,
  effort_limit=PEAK_TORQUE_X6_P20_60,
)

REDUCTION_RATIO_X4_P36_36 = 36.0
ROTOR_INERTIA_X4_P36_36 = 0.3e-4
ARMATURE_X4_P36_36 = reflected_inertia(
  ROTOR_INERTIA_X4_P36_36, REDUCTION_RATIO_X4_P36_36
)
MASS_X4_P36_36 = 0.36
RATED_TORQUE_X4_P36_36 = 10.5
RATED_SPEED_X4_P36_36 = rpm_to_rad(83.0)
PEAK_TORQUE_X4_P36_36 = 34.0
NO_LOAD_SPEED_X4_P36_36 = rpm_to_rad(111.0)
ACTUATOR_X4_P36_36 = ElectricActuator(
  reflected_inertia=ARMATURE_X4_P36_36,
  velocity_limit=NO_LOAD_SPEED_X4_P36_36,
  effort_limit=PEAK_TORQUE_X4_P36_36,
)

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_X8_P20_120 = ARMATURE_X8_P20_120 * NATURAL_FREQ**2
STIFFNESS_X6_P20_60 = ARMATURE_X6_P20_60 * NATURAL_FREQ**2
STIFFNESS_X4_P36_36 = ARMATURE_X4_P36_36 * NATURAL_FREQ**2

DAMPING_X8_P20_120 = 2.0 * DAMPING_RATIO * ARMATURE_X8_P20_120 * NATURAL_FREQ
DAMPING_X6_P20_60 = 2.0 * DAMPING_RATIO * ARMATURE_X6_P20_60 * NATURAL_FREQ
DAMPING_X4_P36_36 = 2.0 * DAMPING_RATIO * ARMATURE_X4_P36_36 * NATURAL_FREQ

P1_ACTUATOR_X8_P20_120 = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_roll_[lr]_joint", r"hip_pitch_[lr]_joint"),
  stiffness=STIFFNESS_X8_P20_120,
  damping=DAMPING_X8_P20_120,
  effort_limit=ACTUATOR_X8_P20_120.effort_limit,
  armature=ACTUATOR_X8_P20_120.reflected_inertia,
)
P1_ACTUATOR_X6_P20_60 = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_yaw_[lr]_joint", r"knee_pitch_[lr]_joint"),
  stiffness=STIFFNESS_X6_P20_60,
  damping=DAMPING_X6_P20_60,
  effort_limit=ACTUATOR_X6_P20_60.effort_limit,
  armature=ACTUATOR_X6_P20_60.reflected_inertia,
)
# Each ankle uses two parallel-coupled X4-P36-36 actuators to control pitch and
# roll. As with the G1 asset, the exact configuration-dependent transmission is
# unavailable, so use a nominal 1:1 mapping and sum both motors' properties.
NUM_ANKLE_MOTORS_PER_LEG = 2
P1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_pitch_[lr]_joint", r"ankle_roll_[lr]_joint"),
  stiffness=STIFFNESS_X4_P36_36 * NUM_ANKLE_MOTORS_PER_LEG,
  damping=DAMPING_X4_P36_36 * NUM_ANKLE_MOTORS_PER_LEG,
  effort_limit=ACTUATOR_X4_P36_36.effort_limit * NUM_ANKLE_MOTORS_PER_LEG,
  armature=ACTUATOR_X4_P36_36.reflected_inertia * NUM_ANKLE_MOTORS_PER_LEG,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.65),
  joint_pos={
    r"hip_pitch_[lr]_joint": -0.1,
    r"knee_pitch_[lr]_joint": 0.3,
    r"ankle_pitch_[lr]_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FOOT_COLLISION_PATTERN = r"^(left|right)_foot[1-4]_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION_PATTERN: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_PATTERN: 1},
  friction={FOOT_COLLISION_PATTERN: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT_COLLISION_PATTERN: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_PATTERN: 1},
  friction={FOOT_COLLISION_PATTERN: (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_COLLISION_PATTERN,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

P1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    P1_ACTUATOR_X8_P20_120,
    P1_ACTUATOR_X6_P20_60,
    P1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_p1_robot_cfg() -> EntityCfg:
  """Get a fresh P1 robot configuration."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=P1_ARTICULATION,
  )


P1_ACTION_SCALE: dict[str, float] = {}
for actuator in P1_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort_limit = actuator.effort_limit
  assert effort_limit is not None
  for name in actuator.target_names_expr:
    P1_ACTION_SCALE[name] = 0.25 * effort_limit / actuator.stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_p1_robot_cfg())
  viewer.launch(robot.spec.compile())
