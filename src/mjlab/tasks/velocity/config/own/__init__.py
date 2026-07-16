"""Own velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import own_flat_env_cfg, own_rough_env_cfg
from .rl_cfg import own_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Own",
  env_cfg=own_rough_env_cfg(),
  play_env_cfg=own_rough_env_cfg(play=True),
  rl_cfg=own_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Own",
  env_cfg=own_flat_env_cfg(),
  play_env_cfg=own_flat_env_cfg(play=True),
  rl_cfg=own_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
