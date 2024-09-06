DQN-Snake Game (CLI-Based)

NOTE: This repo is a proof-of-concept for a learning objective. The code will not be improved. Expect bugs.

Running:
```
poetry install
python3 snek/train_snek.py --help`
```

Overview:
- Implement a DQN agent to play Snake in the terminal.
- Snake moves on a grid, eating food, avoiding walls, and avoiding self-collision (growing snake not yet implemented).
- The goal is to maximize the score (foods eaten with minimal steps).

State Representation:
- Food position.
- Relative positions of the snake's body segments.
- Current direction of movement.
- Boundary position.

Action Space:
- Move Up
- Move Down
- Move Left
- Move Right

Reward Structure:
- see `snek_env.py` NOTE: should be moved to `train_snek.py`

Win Condition:
- no winning. runs forever

File Structure:
- `snek_env.py`: Snake game environment (`SnakeEnv`).
- `dqn_agent.py`: DQN agent (`DQNAgent`).
- `train_snek.py`: Training loop and CLI visualization.

Environment:
- Initialize grid with snake and food.
- Handle movement, collision detection, and resetting.
- Print grid to CLI using ASCII characters (optional).

DQN Agent:
- Use Q-Network for action-value estimation.
- Implement experience replay and batch training.
- Update Q-network based on rewards from environment.

Training Loop:
- Run episodes, update Q-network, print game state to CLI (optional).
- Clear screen after each step to create animation effect (optional).
