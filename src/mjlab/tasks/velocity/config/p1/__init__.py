"""P1 velocity tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import p1_flat_env_cfg
from .rl_cfg import p1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-P1",
  env_cfg=p1_flat_env_cfg(),
  play_env_cfg=p1_flat_env_cfg(play=True),
  rl_cfg=p1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
