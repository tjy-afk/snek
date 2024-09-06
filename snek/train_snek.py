import os
import signal
import argparse
import time
from snek.snek_env import SnakeEnv, MAX_STEPS
from snek.dqn_agent import DQNAgent

# Parameters
SAVE_PATH = "snek_dqn.pth"
LOG_FILE_PATH = "training_stats.txt"

def create_signal_handler(agent, env, log_enabled):
    def signal_handler(sig, frame):
        print("\nTraining interrupted. Saving model and exiting...")
        if agent is not None:
            print(f"Saving model to {SAVE_PATH}...")
            agent.save(SAVE_PATH)
        if log_enabled:
            with open(LOG_FILE_PATH, "a") as log_file:
                log_file.write("Training interrupted with Ctrl+C.\n")
        if env is not None:
            env.close()
        exit(0)
    return signal_handler

def train(render, cli, log_enabled, max_episodes):
    # Initialize the environment
    env = SnakeEnv(grid_size=5)
    state_size = env._get_state().shape[0]
    action_size = 4  # Up, Down, Left, Right

    # Initialize the agent and optionally load the saved model
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    if os.path.exists(SAVE_PATH):
        print(f"Loading existing model from {SAVE_PATH}...")
        agent.load(SAVE_PATH)
        model_status = "Loaded existing model.\n"
    else:
        model_status = "Starting new model.\n"
    print(model_status)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, create_signal_handler(agent, env, log_enabled))
    
    # Open a file for logging stats if logging is enabled
    if log_enabled:
        log_file = open(LOG_FILE_PATH, "a")
        log_file.write(model_status)
    
    episode = 0
    training_in_progress_printed = False
    
    while True:  # Run indefinitely until interrupted or max_episodes is reached
        episode += 1
        state = env.reset()
        env.update_episode(episode)
        total_reward = 0
        done = False

        while not done:
            if render:
                env.render()  # Render the game every episode if the render flag is set
                print(f"Episode: {episode}")
                time.sleep(0.1)  # Slow down the rendering for human observation

            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > agent.batch_size:
                agent.replay()

        # Log and print the episode results with the accumulated reward
        log_line = (f"Episode: {episode:4d} | "
                    f"Score: {env.score:2d}/{env.max_foods:2d} | "
                    f"Steps: {env.steps:3d} | "
                    f"Reward: {env.current_reward:7.2f} | "
                    f"End: {env.end_condition or 'Ongoing':8}")
        
        if log_enabled:
            log_file.write(log_line + "\n")
            log_file.flush()  # Ensure the log is written to disk immediately

        # Print the stats line to CLI every episode if cli flag is set
        if cli:
            print(log_line)

        # Save the model periodically
        if episode % 1000 == 0:
            save_path = f"model_episode_{episode}.pth"
            log_file.write(f"Saving model to {save_path}...")
            agent.save(save_path)

        # Print training in progress message if neither render nor cli flag is set
        if not render and not cli and not training_in_progress_printed:
            print("Training in progress... Use Ctrl+C to end training and update the model.")
            training_in_progress_printed = True

        # Check if max_episodes is reached
        if max_episodes and episode >= max_episodes:
            print(f"Reached max episodes: {max_episodes}. Saving model and exiting...")
            break

    # Save the model at the end of training
    print(f"Saving model to {SAVE_PATH}...")
    agent.save(SAVE_PATH)
    print("Training finished.")
    if log_enabled:
        log_file.write("Training finished.\n")
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Snake DQN agent.")
    parser.add_argument('--render', action='store_true', help="Render the game every episode")
    parser.add_argument('--cli', action='store_true', help="Print stats to CLI every episode")
    parser.add_argument('--log', action='store_true', help="Enable logging to file")
    parser.add_argument('--max-episodes', type=int, help="Set the maximum number of episodes")
    args = parser.parse_args()

    # Ensure that --render and --cli flags cannot be used together
    if args.render and args.cli:
        parser.error("The --render and --cli flags cannot be used together.")

    train(args.render, args.cli, args.log, args.max_episodes)