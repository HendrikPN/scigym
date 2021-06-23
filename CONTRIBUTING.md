# How to contribute

SciGym is a library for scientific [OpenAI gym] reinforcement learning (RL) environments. 
Any RL environment which is of scientific value can be added to this library.
Here, we give a brief description on how to standardize and add your scientific RL environment to SciGym.

## Adding an environment

We recommend that you add any environment also to our website database at [scigym.ai]. The website serves as a curator for scigym environments.
However, your environment does not need to be included to the scigym package to appear on our website. Just log in with your github and upload your repository.

First, standardize your environment according to the OpenAI gym [environment policy].
Then you can directly make a pull request to the SciGym library such that your environment appears within the scigym package.

## How to standardize your environment

There is a [template] that visualizes and explains all the following at https://github.com/hendrikpn/gym-template.

* Create a new repo called e.g. gym-foo, which should also be a PIP package.

* The repo should have at least the following files:
  ```sh
  gym-foo/
    README.md
    setup.py
    gym_foo/
      __init__.py
      envs/
        __init__.py
        foo_env.py
  ```

* `gym-foo/setup.py` should have:

    ```python
    from setuptools import setup

    setup(name='gym_foo',
            version='0.0.1',
            install_requires=['gym']  # And any other dependencies foo needs
    )  
    ```

* `gym-foo/gym_foo/__init__.py` should have:

    ```python
    from gym.envs.registration import register

    register(
        id='foo-v0',
        entry_point='gym_foo.envs:FooEnv',
    )
    ```

* `gym-foo/gym_foo/envs/__init__.py` should have:

    ```python
    from gym_foo.envs.foo_env import FooEnv
    ```

* `gym-foo/gym_foo/envs/foo_env.py` should at least look something like:

    ```python
    import gym
    from gym import error, spaces, utils
    from gym.utils import seeding

    class FooEnv(gym.Env):
        metadata = {'render.modes': ['human']}

        def __init__(self):
        ...
        def step(self, action):
        ...
        def reset(self):
        ...
        def render(self, mode='human', close=False):
        ...
        def close(self):
        ...
    ```

    A short description of each method can also be found in the [template].

## How to add an environment to SciGym

In order to add your standardized environment to the scigym package, follow the following steps.

1. `Fork` scigym.
2. Make a new directory with a title that appropriately captures your scientific problem at `scigym/envs`. For simplicity we call it `scigym/envs/foo` here.
3. Copy your environment file to `scigym/envs/foo/foo_env.py`
4. Add other relevant files to the directory `scigym/envs/foo` such as a README.md.
5. Add any dependencies that your environment has to our `setup.py` in `extras`:

    ```python
    extras = {
    'foo': ['Dependencies'],
    }
    ```
6. Additionally, in order to allow automated unit testing of your environment, add any environment specific dependencies into the file `requirements.txt` in the root directory of the package.
7. Register your environment `FooEnv` at `scigym/envs/__init__.py`. Take note, that if the environment is non-deterministic then this attribute should be set to true, as determinism is verified by the unit tests. :

    ```python
    register(
        id='foo-v0',
        entry_point='scigym.envs.foo:FooEnv',
        nondeterministic = False
    )
    ```

8. Import your environment in `scigym/envs/foo/__init__.py`:

    ```python
    from scigym.envs.foo.foo_env import FooEnv
    ```

9. Test your environment. Add your environment to the list of environments in `test_random_rollout()` in `scigym/envs/tests/test_envs.py`. Run `python -m pytest` in the root folder to test your environment. 
10. If your environment requires unit tests not covered by `scigym/envs/tests/test_envs.py` and `scigym/envs/tests/test_determinism.py`, then add an environment specific unit test into the folder `scigym/envs/tests/`. Scigym uses `pytest` for all unit tests.

11. Make a notification that a new environment has been included in the README.md under "What's new". 
12. If your environment should be registered under a licence other than MIT, publish your licence statement in the LICENCE file under "Special Environments".
13. Create a branch `environment/foo` and commit the changes there.
14. Create a pull request to the master branch or a new branch at https://github.com/hendrikpn/scigym.


  [OpenAI gym]: https://github.com/openai/gym
  [scigym.ai]: https://scigym.ai
  [environment policy]: https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym
  [template]: https://hendrikpn.github.io/gym-template/
