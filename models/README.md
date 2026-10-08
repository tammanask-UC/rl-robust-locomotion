# models/

Copy the 12 final models from `MyDrive/RL_Part2/rl_part2_final_results/models/`:

ppo_nominal_seed{0,1,2}.zip, ppo_randomized_seed{0,1,2}.zip,
sac_nominal_seed{0,1,2}.zip, sac_randomized_seed{0,1,2}.zip

Do **not** copy the replay-buffer `.pkl` files or the `checkpoints/` folder (too large).

Load a model with:

    from stable_baselines3 import PPO, SAC
    model = PPO.load("models/ppo_randomized_seed1.zip")
