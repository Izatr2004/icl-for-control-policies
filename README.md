# icl for control policies

This project studies whether a frozen transformer can infer a held-out
control policy from a few demonstrations.

The initial experiment uses ManiSkill `PushT-v1` with the Panda arm and stick
end effector. Start with numerical state observations and a single controller
mode shared across teachers. Validate a released PPO expert and its
demonstrations before building teacher-level splits and testing whether
correct context improves prediction and rollout performance compared with
empty and wrong-teacher context. Follow with RGB and robot proprioception.

The saved course proposal still describes Gym-PushT.

Full-state BC teachers (Nathan): notebooks, code, and the running log of runs and
decisions are in [state_bc/README.md](state_bc/README.md).

See [CODEX_CONTEXT.md](CODEX_CONTEXT.md) for the research handoff and a
starter prompt for new Codex sessions.
