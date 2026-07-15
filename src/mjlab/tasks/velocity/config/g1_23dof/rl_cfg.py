"""RL configuration for Unitree G1-23DOF velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_23dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1-23DOF velocity task."""
  return RslRlOnPolicyRunnerCfg(
    #根据观测产生动作
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True, #归一化不同量纲的观测
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0, #训练初期保持较强探索
        "std_type": "scalar",
      },
    ),
    #评价当前状态的长期好坏
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, #价值损失权重
      use_clipped_value_loss=True, #限制 Critic 单次更新幅度
      clip_param=0.2, #裁剪价值损失（限制新旧策略变化过大）
      entropy_coef=0.01, #熵损失权重（鼓励策略探索，防止过早收敛）
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive", #根据 KL 散度动态调整学习率
      gamma=0.99, #折扣因子（未来奖励的衰减系数，越接近1表示越重视未来奖励）
      lam=0.95, #GAE参数（用于计算优势函数的平滑因子，越接近1表示越平滑）
      desired_kl=0.01, #期望KL散度
      max_grad_norm=1.0, #梯度裁剪
    ),
    experiment_name="g1_23dof_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
