# Soft Actor Critic
This is a simple, self-contained pytorch implementation of the Reinforcement Learning agent Soft Actor Critic.

The *actor.py* and *critic.py* files contain the networks of the two agents. The *SAC.py* then implements the SAC algorithm. The *replay_buffer_bounded.py* implements our replay buffer, which has a limited capacity, so after a certain amount of training steps the agent learns from recent experiences rather than the first explorative steps.

The agent works well on simple environments but needs some reward tuning for more complicated environments.

To run the project, select a continuous enviroment of your choice in sac_run_simple_env.py (default is pendulum) and run the file.
Here is some example of the agent solving the pendulum environemnt:
0 epochs | 50 epochs| 150 epochs
--- | --- | ---
<img src="pendulum_e_0.gif" width="150" height="150"/> | <img src="pendulum_e_50.gif" width="150" height="150"/> |<img src="pendulum_e_150.gif" width="150" height="150"/>
