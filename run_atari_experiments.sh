#!/bin/bash
# Use these commands to run the experiments on the respective model configurations.

# Run experiment for all noop policy in Figure 6 (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "noop" --explore 0 --cuda no --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "noop" --explore 0 --cuda no --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "noop" --explore 0 --cuda no --seed 0

# Run experiment for random policy in Figure 6 (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "random" --explore 0 --cuda no --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "random" --explore 0 --cuda no --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "random" --explore 0 --cuda no --seed 0

# For the 1xDeep 55k parameter model in Figure 6 with inference time equal to 1 step, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 1 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 1 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 1 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 1 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 1 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 1 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 1 --noops 1 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 1 --noops 1 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 1 --noops 1 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 1 --noops 1 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 1 --noops 1 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 1 --noops 1 --seed 0

# For the 2xDeep 70K parameter model in Figure 6 with inference time equal to 3 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 3 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 3 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 3 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 3 --noops 3 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 3 --noops 3 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 3 --noops 3 --seed 0


# For the 1xDeep 1M parameter model in Figure 6 with inference time equal to 5 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 5 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 5 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 5 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 5 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 5 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 5 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 5 --noops 5 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 5 --noops 5 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 5 --noops 5 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 5 --noops 5 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 5 --noops 5 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 5 --noops 5 --seed 0

# For the 2xDeep 1M parameter model in Figure 6 with inference time equal to 29 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 29 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 29 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 29 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 29 --noops 29 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 29 --noops 29 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 29 --noops 29 --seed 0


# For the 1xDeep 10M parameter model in Figure 6 with inference time equal to 8 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 8 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 8 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 8 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 8 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 8 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 8 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 8 --noops 8 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 8 --noops 8 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 8 --noops 8 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 8 --noops 8 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 8 --noops 8 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 8 --noops 8 --seed 0

# For the 2xDeep 10M parameter model in Figure 6 with inference time equal to 47 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 47 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 47 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 47 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 47 --noops 47 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 47 --noops 47 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 47 --noops 47 --seed 0


# For the 1xDeep 100M parameter model in Figure 6 with inference time equal to 70 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 70 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 70 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 70 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 70 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 70 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 70 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 70 --noops 70 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 70 --noops 70 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 70 --noops 70 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 70 --noops 70 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 70 --noops 70 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 70 --noops 70 --seed 0

# For the 2xDeep 100M parameter model in Figure 6 with inference time equal to 80 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 80 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 80 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 80 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 80 --noops 80 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 80 --noops 80 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 80 --noops 80 --seed 0


# For the 1xDeep 1B parameter model in Figure 6 with inference time equal to 599 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 599 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 599 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 599 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 599 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 599 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 599 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 599 --noops 599 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 599 --noops 599 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 599 --noops 599 --seed 0
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 599 --noops 599 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "rainbow" --cuda yes --delay 599 --noops 599 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "rainbow" --cuda yes --delay 599 --noops 599 --seed 0


# For the 2xDeep 1B parameter model in Figure 6 with inference time equal to 750 steps, simulate asynchronous interaction (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 750 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 750 --noops 0 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 750 --noops 0 --seed 0
# Simulate sequential interaction for the same model (our experiments included seed = 0, 1, and 2)
python3 simulate_delay_atari.py --game "BoxingNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 750 --noops 750 --seed 0
python3 simulate_delay_atari.py --game "KrullNoFrameSkip-v4" --num_actions 18 --model "dqn" --cuda yes --delay 750 --noops 750 --seed 0
python3 simulate_delay_atari.py --game "NameThisGameNoFrameSkip-v4" --num_actions 6 --model "dqn" --cuda yes --delay 750 --noops 750 --seed 0