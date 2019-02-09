from setuptools import setup, find_packages
import sys, os.path

# Don't import scigym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scigym'))
from version import VERSION

# Environment-specific dependencies.
extras = {
    'test_envs': ['PyOpenGL'],
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

setup(name='scigym_pkg',
      version=VERSION,
      description='SciGym -- The OpenAI Gym for Science: A platform for your scientific reinforcement learning problem.',
      url='https://github.com/HendrikPN/scigym',
      author='HendrikPN',
      author_email='hendrik.poulsen-nautrup@uibk.ac.at',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('scigym')],
      zip_safe=False,
      install_requires=[
          'gym',
      ],
      extras_require=extras,
      package_data={
      },
      tests_require=[],
)
