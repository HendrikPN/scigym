from gym.envs.registration import registry, register, make, spec


register(
    id='surfacecode-decoding-v0',
    entry_point='scigym.envs.quantum_physics.quantum_computing.surfacecode_decoding:SurfaceCodeEnv',
    nondeterministic=True
)