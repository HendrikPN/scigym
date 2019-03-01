from gym.envs.registration import registry, register, make, spec


# Test Environment
# ----------------------------------------

register(
    id='CartPole-test-v0',
    entry_point='scigym.envs.test_envs:CartPoleEnv',
)

register(
    id='surfacecode-v0',
    entry_point='scigym.envs.quantum_physics.quantum_computing.surfacecode_decoding:SurfaceCodeEnv',
)