from gym.envs.registration import registry, register, make, spec

register(
    id='teleportation-v0',
    entry_point="scigym.envs.quantum_physics.quantum_computing.teleportation:TeleportationEnv",
    nondeterministic=True
)

register(
    id='entangled-ions-v0',
    entry_point='scigym.envs.quantum_physics.quantum_information.entangled_ions:EntangledIonsEnv',
)

register(
    id='toricgame-v0',
    entry_point="scigym.envs.quantum_physics.quantum_computing.toricgame:ToricGameEnv",
    nondeterministic=True
)
