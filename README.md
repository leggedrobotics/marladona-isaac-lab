# MARLadona 
This repository contains the multi-agent training environment for the [MARLadona - Towards Cooperative Team Play Using Multi-Agent
Reinforcement Learning](https://arxiv.org/pdf/2409.20326) Paper.

The open-source version of the MARL soccer environment is built on top of [IsaacLab](https://github.com/isaac-sim/IsaacLab) and based on the [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate.git) 

This repository contains the multi-agent soccer environment `isaaclab_marl` and a heavily modified [rsl_rl](https://github.com/leggedrobotics/rsl_rl) training pipeline implemented in `rsl_marl`. The original implementation and paper results are based on Isaac Gym. This migration effort was made due to Isaac Gym's deprecation.         

<figure>
  <figcaption><em>Cooperative Gameplay Like Never Before! (Isaaclab)</em></figcaption>
  <img src="docs/gifs/isaaclab.gif" alt="Isaaclab Play" title="Typical Gameplay in Isaac Lab">
</figure>


<figure>
  <figcaption><em>This Single Policy Adapts to Any Team Setup! (Isaac Gym)</em></figcaption>
  <img src="docs/gifs/isaacgym.gif" alt="Isaaclab Play" title="Higher Agent Number (Isaac Gym)">
</figure>

## Installation

- Install Isaac Sim and Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: HTTPS
git clone https://github.com/leggedrobotics/marladona-isaac-lab.git

# Option 2: SSH
git clone git@github.com:leggedrobotics/marladona-isaac-lab.git
```

```bash
# Enter the repository
cd marladona-isaac-lab
```

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/isaaclab_marl
python -m pip install -e source/rsl_marl
```

- Verify that the extension is correctly installed by running the following command:

```bash
python scripts/rsl_rl/train.py --task=Isaac-Soccer-v0 
```

We assumed `wks_logs` to be our default root log folder for all our scripts. An example policy is already provided there. You can test its performance by running the following command:

```bash
python scripts/rsl_rl/play.py --task=Isaac-Soccer-Play-v0 --experiment_name=00_example_policies --load_run=24_09_28_11_56_41_3v3 
```
Note: The number of agents can be configured via the `SoccerMARLEnvPlayCfg` class.

### Useful Visualization Tools ### 

## Trajectory Analyser ## 
![Trajectory Analyser](docs/gifs/3v3_traj.gif) 

The framework provides a convenient GUI to visualize and compare policy behavior across many experiments. The trajectories are collected periodically during training on the evaluation environments, which is about 15% of the total environment. In these environments, the adversaries are configured to use a simple heuristic bot as a controller to increase reproducibility and also provide a standardized resistance to our trainees. Furthermore, all randomizations regarding the team size and initial position are fixed. This makes qualitative comparisons of behavior between different checkpoints and experiments much easier. 

To start the trajectory analyser, simply run the following command: 
```bash
python scripts/traj_analyser.py 
```

<img src="docs/pngs/traj_analyser_gui.png" alt="GUI"/>

You can select the experiment_folder and run name from the dropdown box on the left. This will automatically update the sliders in the middle. The sliders allow you to filter the trajectories according to the team configuration, and you can easily iterate over all checkpoints and all environments with the given team configuration.  

Furthermore, the GUI also supports storing highlights, which can be managed via the Add and Delete buttons on the right side.      

The GUI assumes all logs are stored inside the `wks_logs` folder. It selects only the experiment folder prefix with digits, e.g., `00_example_policies`. Make sure all runs contain a non-empty `eval_traj` folder. This should be the case for all training runs that have finished the initialization. 

Note: The GUI application is built using pyqtgraph and PyQt5, so double-check your pip package version to see if the dependencies are not already auto-resolved by the setup.py.

## Value Function Visualizer ## 

![Value Functions](docs/gifs/value_function.gif)

The value function visualizer provides additional insight into the agent's intention. You can enable or disable the visualization via the `VISUALIZE_VALUE_FUN` flag in the play.py script.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```