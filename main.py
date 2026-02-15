import gym
import torille.envs
import argparse
from src.mocap2combat.utils.config import PPO_CONFIG
from src.mocap2combat.trainer.ppo_trainer import PPOTrainer


def make_env(env_name="Toribash-DestroyUke-v1", render: bool = True):
    env = gym.make(env_name, disable_env_checker=True)
    # Toribash-gym uses draw toggle (not gym render_mode), so gate it
    env.set_draw_game(bool(render))
    return env



def train_ppo():
    env = make_env(render=False)
    PPO_CONFIG['state_dim'] = env.observation_space.shape[0]
    PPO_CONFIG['action_dim'] = env.action_space.shape[0]
    trainer = PPOTrainer(env, PPO_CONFIG)
    trainer.train()


def random_agent(env):
    obs = env.reset()
    done = False
    score = 0
    print("obs space:", env.observation_space.shape)
    print("action space:", env.action_space.shape)
    while not done:
        action = env.action_space.sample()
        print("Taking random action:", action)
        obs, reward, done, info = env.step(action)
        score += reward
    print("Episode finished with score:", score)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["random", "ppo-training"],
        help="Which script to run",
    )

    args = parser.parse_args()

    if args.mode == "random":
        random_agent(make_env())

    elif args.mode == "ppo-training":
        train_ppo()

if __name__ == "__main__":
    main()