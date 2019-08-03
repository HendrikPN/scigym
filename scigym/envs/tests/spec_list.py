
from scigym import envs

def should_skip_env_spec_for_tests(spec):
    # Currently all scigym environments are subject to testing.
    # Future environments which should be handled seperately can be added here
    return False

spec_list = [spec for spec in sorted(envs.registry.all(), key=lambda x: x.id) if spec.entry_point[:6] == "scigym" and not should_skip_env_spec_for_tests(spec)]


