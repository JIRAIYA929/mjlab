"""P1 双足机器人平地速度跟踪环境配置。"""

import math

from mjlab.asset_zoo.robots import P1_ACTION_SCALE, get_p1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# 足端 site 用于读取足端位置、速度以及离地高度。
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
# 每只脚由 4 个球形碰撞点近似，用于地面接触和摩擦随机化。
_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
)
# 一个完整左右步态周期为 0.7 秒，对低速稳定行走较为保守。
_GAIT_PERIOD = 0.7


def p1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """创建以稳定性为优先的 P1 平地速度跟踪任务。"""
  # 复用通用速度任务
  cfg = make_velocity_env_cfg()

  # 场景只包含一台 P1；本任务只训练平地行走，因此关闭崎岖地形生成器。
  cfg.scene.entities = {"robot": get_p1_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 平地不需要 pelvis 周围的密集地形扫描，因此同时删除对应传感器及观测。
  # foot_height_scan 仍需保留：critic 观测、抬脚高度和落脚高度奖励都会使用它。
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # 在左右脚 site 周围布置半径 4 cm 的 8 点圆环，测量脚底相对地面高度。
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in _FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=8)

  # 双脚—地面接触：分别统计左右脚的合接触力、接触时间和腾空时间。
  # primary 使用 ankle_roll 子树，因此会包含该脚下方的 4 个碰撞点。
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(ankle_roll_l_link|ankle_roll_r_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # 机器人自碰撞
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="pelvis_link",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="pelvis_link",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # 非脚部—地面接触：pelvis、髋、大腿/膝和踝 pitch 碰地均视为摔倒。
  body_ground_cfg = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=(
        "pelvis_collision",
        r"hip_.*_collision",
        r"knee_pitch_.*_collision",
        r"ankle_pitch_.*_collision",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # 将 P1 专用接触传感器追加到通用场景已有的足端高度传感器
  cfg.scene.sensors = cfg.scene.sensors + (
    feet_ground_cfg,
    self_collision_cfg,
    body_ground_cfg,
  )

  # 策略输出 12 维归一化关节位置动作，
  # P1_ACTION_SCALE 根据各电机的峰值力矩和 PD 刚度分别计算，
  # 最终目标位置为默认站姿加缩放后的策略动作。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = P1_ACTION_SCALE

  # 相机跟随 pelvis_link，并使用适合观察完整双足步态的距离和俯视角。
  cfg.viewer.body_name = "pelvis_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -8.0

  # 速度指令
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # 每 4～8 秒重新采样一次目标速度，给策略足够时间完成过渡。
  twist_cmd.resampling_time_range = (4.0, 8.0)
  # 20% 环境训练原地站立；其余环境包含定向、纯前进和一般速度指令。
  twist_cmd.rel_standing_envs = 0.2
  twist_cmd.rel_heading_envs = 0.2
  twist_cmd.rel_forward_envs = 0.4
  twist_cmd.heading_control_stiffness = 0.5
  # 初始速度范围刻意较小，后续由课程学习逐步扩展。
  twist_cmd.ranges.lin_vel_x = (-0.25, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.1, 0.1)
  twist_cmd.ranges.ang_vel_z = (-0.25, 0.25)
  # 将速度指令箭头显示在 pelvis 上方，避免被模型遮挡。
  twist_cmd.viz.z_offset = 0.8

  # 观测空间和步态相位

  # 通用配置已经定义观测：
  # - actor：IMU 角速度、投影重力、关节位置/速度、上一动作、速度指令和相位；
  # - critic：在 actor 基础上增加真实基座线速度、足端高度/接触/力等特权信息。
  # actor 不使用真实基座线速度，有利于后续只依赖机载传感器进行实机部署。
  # actor 和 critic 使用相同的 sin/cos 周期相位，帮助策略学习左右交替步态。
  cfg.observations["actor"].terms["phase"].params["period"] = _GAIT_PERIOD
  cfg.observations["critic"].terms["phase"].params["period"] = _GAIT_PERIOD

  # 重置、外力扰动和域随机化事件
  # 重置根节点时随机化位置、朝向和初速度。roll/pitch 范围较小，保证训练
  # 初期仍以站稳和迈步为主；yaw 覆盖整圆，避免策略依赖固定世界朝向。
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (0.01, 0.03),
    "roll": (-0.03, 0.03),
    "pitch": (-0.03, 0.03),
    "yaw": (-math.pi, math.pi),
  }
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (-0.05, 0.05),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.1, 0.1),
  }

  # 在默认站姿附近轻微随机化关节位置和速度，提高对初始化误差的容忍度。
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)

  # 每 8～12 秒施加一次温和速度扰动，训练基础抗扰恢复能力。
  cfg.events["push_robot"].interval_range_s = (8.0, 12.0)
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.15, 0.15),
    "pitch": (-0.15, 0.15),
    "yaw": (-0.2, 0.2),
  }

  # 随机化 8 个脚底碰撞点的滑动摩擦系数，左右脚共享同一次采样值。
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_NAMES
  cfg.events["foot_friction"].params["ranges"] = (0.5, 1.0)
  # 模拟关节编码器零位偏差。
  cfg.events["encoder_bias"].params["bias_range"] = (-0.01, 0.01)
  # 模拟 pelvis 内部载荷和装配误差导致的质心偏移。
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_link",)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.015, 0.015),
  }

  # 速度跟踪奖励
  # 线速度是主要任务目标，权重高于角速度。
  # std 是指数核的容差尺度：数值越小，偏离指令后奖励衰减越快。
  cfg.rewards["track_linear_velocity"].weight = 1.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.4
  cfg.rewards["track_angular_velocity"].weight = 1.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.5
  # 只轻微惩罚 roll/pitch 角速度，避免妨碍正常步态摆动。
  cfg.rewards["track_angular_velocity"].params["xy_weight"] = 0.05

  # 躯干姿态和关节姿态奖励

  # 使用连续的姿态平方惩罚替换容易饱和的 upright 正奖励
  # pelvis 越倾斜，惩罚越大，使策略始终获得恢复竖直姿态的梯度。
  cfg.rewards.pop("upright")
  cfg.rewards["body_orientation_l2"] = RewardTermCfg(
    func=mdp.body_orientation_l2,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis_link",)),
    },
  )

  # variable_posture 根据指令速度切换容差。站立时强约束默认姿态；行走时
  # 放宽 hip pitch 和 knee，保留较紧的 hip/ankle roll 以减少左右摇摆。
  # 每项 std 越小，该关节偏离默认站姿的代价越大。
  cfg.rewards["pose"].weight = 0.5
  cfg.rewards["pose"].params["walking_threshold"] = 0.1
  cfg.rewards["pose"].params["running_threshold"] = 1.1
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.25,
    r".*hip_roll.*": 0.12,
    r".*hip_yaw.*": 0.1,
    r".*knee_pitch.*": 0.35,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.08,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.12,
    r".*knee_pitch.*": 0.4,
    r".*ankle_pitch.*": 0.18,
    r".*ankle_roll.*": 0.1,
  }

  # 抑制 pelvis 横滚/俯仰角速度以及整机角动量，降低上身晃动。
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_link",)
  cfg.rewards["body_ang_vel"].weight = -0.1
  cfg.rewards["angular_momentum"].weight = -0.005
  # 限制相邻控制步动作突变和关节加速度，减少高频抖动。
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-1.0e-7,
  )
  # 加大软关节限位惩罚，并将名称改得更明确。
  joint_pos_limits = cfg.rewards.pop("dof_pos_limits")
  joint_pos_limits.weight = -10.0
  cfg.rewards["joint_pos_limits"] = joint_pos_limits

  # 不单独奖励长腾空时间，防止策略通过跳跃获得收益。
  cfg.rewards["air_time"].weight = 0.0
  # 左右脚相位差为通用配置中的 0.5；threshold=0.6 形成约 20% 双支撑，
  # 比完全无重叠的交替支撑更适合低速稳定行走。
  cfg.rewards["foot_gait"].weight = 0.5
  cfg.rewards["foot_gait"].params["period"] = _GAIT_PERIOD
  cfg.rewards["foot_gait"].params["threshold"] = 0.6
  cfg.rewards["foot_gait"].params["command_threshold"] = 0.1
  # 目标抬脚高度为 6 cm，同时约束运动过程和落脚时的峰值高度。
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  # 脚与地面接触时惩罚水平速度，减少拖脚和打滑。
  cfg.rewards["foot_slip"].weight = -0.2
  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = _FOOT_SITE_NAMES
  # 零速度指令下要求回到默认站姿，避免原地踏步。
  cfg.rewards["stand_still"].weight = -2.0
  # 对超过 10 N 的机器人自碰撞施加惩罚。
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # 平面没有边界，因此删除崎岖地形任务中的越界终止。
  cfg.terminations.pop("out_of_terrain_bounds", None)
  # pelvis 倾斜超过 50°视为倾倒。
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(50.0)},
  )
  # pelvis 高度低于 0.4 m 说明机器人已经明显下蹲失稳或倒地。
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.4},
  )
  # 除脚底外的主要碰撞体以超过 10 N 的力接触地面时立即结束回合。
  cfg.terminations["illegal_body_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": body_ground_cfg.name,
      "force_threshold": 10.0,
    },
  )

  # 平地任务不需要地形难度课程，只逐步扩大速度范围。step 按环境控制步
  # 计数；每次 PPO 迭代采样 24 步，因此 1500*24 约对应第 1500 次迭代。
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    # 阶段 1：先学习站立、低速前进和小幅转向。
    {
      "step": 0,
      "lin_vel_x": (-0.25, 0.5),
      "lin_vel_y": (-0.1, 0.1),
      "ang_vel_z": (-0.25, 0.25),
    },
    # 阶段 2：加入更明显的横移、后退和转向。
    {
      "step": 3000 * 24,
      "lin_vel_x": (-0.4, 0.7),
      "lin_vel_y": (-0.15, 0.15),
      "ang_vel_z": (-0.4, 0.4),
    },
    # 阶段 3：最终基本行走范围，最高前进速度为 0.8 m/s。
    {
      "step": 6000 * 24,
      "lin_vel_x": (-0.5, 0.8),
      "lin_vel_y": (-0.2, 0.2),
      "ang_vel_z": (-0.5, 0.5),
    },
  ]

  # 平地接触规模较小，使用中等 CCD 迭代和接触匹配容量即可。
  # nconmax=None 让后端按实际模型自动决定接触缓冲区。
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = None

  if play:
    # 推理时近似取消超时，关闭观测噪声、外力扰动和课程更新。
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    # play 直接开放训练课程的最终速度范围，便于完整测试策略。
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
