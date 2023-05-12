from rl_tools.agent import QAgent, UCBQAgent
from rl_tools.env import Map2DStatic, Map2DDynamic
import numpy as np


def main():
    # env = Map2DStatic(
    #     shape=(10, 10),
    #     start=(0, 0),
    #     goal=(9, 9),
    #     iterations=100,
    # )

    env = Map2DDynamic(
        shape=(10, 10),
        iterations=100
    )

    # Create agent
    agent = UCBQAgent(
        alpha=0.1,
        gamma=0.9,
        c=0.1,
    )

    ep_rewards = []
    stats = {"ep": [], "mean": [], "max": [], "min": []}

    # Train agent
    for ep in range(1, 10_001):
        state = env.reset()
        done = False

        ep_reward = 0

        while not done:
            action = agent.get_action(state, env.get_actions(state))
            next_state, reward, done = env.step(action)
            ep_reward += reward

            agent.update(state, action, reward, next_state, done)
            state = next_state
            if ep % 1_000:
                env.render()

        ep_rewards.append(ep_reward)

        if ep % 1_000 == 0:
            stats["ep"].append(ep)
            stats["mean"].append(round(np.mean(ep_rewards[-1_000:]), 2))
            stats["max"].append(round(max(ep_rewards[-1_000:]), 2))
            stats["min"].append(round(min(ep_rewards[-1_000:]), 2))

            print(f"Episode: {stats['ep'][-1]: 8} Mean: {stats['mean'][-1]: 4} Max: {stats['max'][-1]: 4} Min: {stats['min'][-1]: 4}")


    # # Test agent
    # import pygame
    #
    # state = env.reset()
    # env.render()
    # done = False
    # while not done:
    #     action = agent.get_action(state, env.get_actions(state))
    #     next_state, reward, done = env.step(action)
    #     state = next_state
    #     env.render()
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    # pygame.quit()


if __name__ == "__main__":
    main()
