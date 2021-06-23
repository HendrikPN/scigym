**Status:** Development (expect bug fixes, minor updates and new
environments)

<a href="https://unitary.fund/">
    <img src="https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge"
    />
</a>

# SciGym

<a href="https://www.scigym.net">
    <img src="https://raw.githubusercontent.com/HendrikPN/scigym/master/assets/scigym-logo.png" width="120px" align="bottom"
    />
</a>

**SciGym is a curated library for reinforcement learning environments in science.**
This is the `scigym` open-source library which gives you access to a standardized set of science environments.
Visit our webpage at [scigym.ai]. This website serves as a open-source database for science environments: A port where science and reinforcement learning meet.

<a href="https://travis-ci.org/HendrikPN/scigym">
    <img src="https://travis-ci.org/HendrikPN/scigym.svg?branch=master" align="bottom"
    />
</a>

[See What's New section below](#whats-new)

## Basics

This project is in line with the policies of the [OpenAI gym]:

There are two basic concepts in reinforcement learning: the environment
(namely, the outside world) and the agent (namely, the algorithm you are
writing). The agent sends `actions` to the environment, and
the environment replies with `observations` and
`rewards` (that is, a score).

The core `gym` interface is [Env], which is the unified
environment interface. There is no interface for agents; that part is
left to you. The following are the `Env` methods you should know:

* `reset(self)`: Reset the environment's state. Returns `observation`.
* `step(self, action)`: Step the environment by one timestep. Returns `observation`, `reward`, `done`, `info`.
* `render(self, mode='human', close=False)`: Render one frame of the environment. The default mode will do something human friendly, such as pop up a window. Passing the `close` flag signals the renderer to close any such windows.

## Installation

There are two main options for the installation of `scigym`:

#### (a) minimal install (recommended)

This method allows you to install the package with no environment specific dependencies, and later add the dependencies for specific environments as you need them.

You can perform a minimal install of `scigym` with:

  ```sh
  pip install scigym
  ```
or
  ```sh
  git clone https://github.com/hendrikpn/scigym.git
  cd scigym
  pip install -e .
  ```

To later add the dependencies for a particular `environment_name`, run the following command:

  ```sh
  pip install scigym[environment_name]
  ```
or from the folder containing `setup.py`
  ```sh
  pip install -e .[environment_name]
  ```

#### (b) full install

This method allows you to install the package, along with all dependencies required for all environments. Be careful, scigym is growing, and this method may install a large number of packages. To view all packages that will be installed during a full install, see the `requirements.txt` file in the root directory. If you wish to perform a full installation you can run:

  ```sh
  pip install scigym['all']
  ```
or
  ```sh
  git clone https://github.com/hendrikpn/scigym.git
  cd scigym
  pip install -e .['all']
  ```

## Available Environments

At this point we have the following environments available for you to play with:

- [`teleportation`](https://github.com/HendrikPN/scigym/tree/master/scigym/envs/quantum_physics/quantum_computing/teleportation)
- [`entangled-ions`](https://github.com/HendrikPN/scigym/tree/master/scigym/envs/quantum_physics/quantum_information/entangled_ions)

## What's New

- 2021-06-16 Added the [Toric Game](https://github.com/HendrikPN/scigym/tree/master/scigym/envs/quantum_physics/quantum_computing/toricgame) environment
- 2021-06-09 Added [entangled-ions](https://github.com/HendrikPN/scigym/tree/master/scigym/envs/quantum_physics/quantum_information/entangled_ions) environment.
- 2021-06-08 This is `scigym` version 0.0.3! Now compatible with gym version 0.18.0
- 2019-10-10 [scigym.ai] is online!
- 2019-08-30 This is `scigym` version 0.0.2!
- 2019-08-30 `scigym` is now available as a package on [PyPI](https://pypi.org/project/scigym/).
- 2019-08-06 Added [Travis-CI](https://travis-ci.org/HendrikPN/scigym).
- 2019-08-06: Added [teleportation](https://github.com/HendrikPN/scigym/tree/master/scigym/envs/quantum_physics/quantum_computing/teleportation) environment.
- 2019-07-21: Added standardized unit testing for all scigym environments.
- 2019-03-04: Added <a href="https://github.com/R-Sweke/gym-surfacecode">surfacecode</a> environment.
- 2019-02-09: Initial commit. Hello world :)

  [image]: https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge
  [OpenAI gym]: https://github.com/openai/gym
  [scigym.ai]: https://www.scigym.net
  [Env]: https://github.com/openai/gym/blob/master/gym/core.py
