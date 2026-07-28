"""P1 biped robot constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import reflected_inertia
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

REDUCTION_RATIO_X8_P20_120 = 19.612
ROTOR_INERTIA_X8_P20_120 = 1.5e-4
ARMATURE_X8_P20_120 = reflected_inertia(
  ROTOR_INERTIA_X8_P20_120, REDUCTION_RATIO_X8_P20_120
)
PEAK_TORQUE_X8_P20_120 = 120.0

REDUCTION_RATIO_X6_P20_60 = 19.612
ROTOR_INERTIA_X6_P20_60 = 0.66e-4
ARMATURE_X6_P20_60 = reflected_inertia(
  ROTOR_INERTIA_X6_P20_60, REDUCTION_RATIO_X6_P20_60
)
PEAK_TORQUE_X6_P20_60 = 60.0

REDUCTION_RATIO_X4_P36_36 = 36.0
ROTOR_INERTIA_X4_P36_36 = 0.3e-4
ARMATURE_X4_P36_36 = reflected_inertia(
  ROTOR_INERTIA_X4_P36_36, REDUCTION_RATIO_X4_P36_36
)
PEAK_TORQUE_X4_P36_36 = 34.0

STIFFNESS_HIP_PITCH_KNEE = 200.0
DAMPING_HIP_PITCH_KNEE = 6.0
STIFFNESS_HIP_ROLL_YAW = 150.0
DAMPING_HIP_ROLL_YAW = 6.0
STIFFNESS_ANKLE_MOTOR = 100.0
DAMPING_ANKLE_MOTOR = 6.0

P1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_pitch_[lr]_joint",),
  stiffness=STIFFNESS_HIP_PITCH_KNEE,
  damping=DAMPING_HIP_PITCH_KNEE,
  effort_limit=PEAK_TORQUE_X8_P20_120,
  armature=ARMATURE_X8_P20_120,
)
P1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(r"knee_pitch_[lr]_joint",),
  stiffness=STIFFNESS_HIP_PITCH_KNEE,
  damping=DAMPING_HIP_PITCH_KNEE,
  effort_limit=PEAK_TORQUE_X6_P20_60,
  armature=ARMATURE_X6_P20_60,
)
P1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_roll_[lr]_joint",),
  stiffness=STIFFNESS_HIP_ROLL_YAW,
  damping=DAMPING_HIP_ROLL_YAW,
  effort_limit=PEAK_TORQUE_X8_P20_120,
  armature=ARMATURE_X8_P20_120,
)
P1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(r"hip_yaw_[lr]_joint",),
  stiffness=STIFFNESS_HIP_ROLL_YAW,
  damping=DAMPING_HIP_ROLL_YAW,
  effort_limit=PEAK_TORQUE_X6_P20_60,
  armature=ARMATURE_X6_P20_60,
)
# Each ankle uses two parallel-coupled X4-P36-36 motors.
NUM_ANKLE_MOTORS_PER_LEG = 2
STIFFNESS_ANKLE_JOINT = STIFFNESS_ANKLE_MOTOR * NUM_ANKLE_MOTORS_PER_LEG
DAMPING_ANKLE_JOINT = DAMPING_ANKLE_MOTOR * NUM_ANKLE_MOTORS_PER_LEG
P1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(r"ankle_pitch_[lr]_joint", r"ankle_roll_[lr]_joint"),
  stiffness=STIFFNESS_ANKLE_JOINT,
  damping=DAMPING_ANKLE_JOINT,
  effort_limit=PEAK_TORQUE_X4_P36_36 * NUM_ANKLE_MOTORS_PER_LEG,
  armature=ARMATURE_X4_P36_36 * NUM_ANKLE_MOTORS_PER_LEG,
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
    P1_ACTUATOR_HIP_PITCH,
    P1_ACTUATOR_KNEE,
    P1_ACTUATOR_HIP_ROLL,
    P1_ACTUATOR_HIP_YAW,
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


P1_ACTION_SCALE = {
  r"hip_pitch_[lr]_joint": 0.25,
  r"knee_pitch_[lr]_joint": 0.25,
  r"hip_roll_[lr]_joint": 0.1,
  r"hip_yaw_[lr]_joint": 0.1,
  r"ankle_pitch_[lr]_joint": 0.1,
  r"ankle_roll_[lr]_joint": 0.1,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_p1_robot_cfg())
  viewer.launch(robot.spec.compile())
