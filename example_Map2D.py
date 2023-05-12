from rl_tools.agent import QAgent, UCBQAgent
from rl_tools.env import Map2DStatic


def main():
    # Create environment
    env = Map2DStatic(
        shape=(10, 10),
        start=(0, 0),
        goal=(9, 9),
        iterations=100,
    )

    # Create agent
    agent = UCBQAgent(
        alpha=0.1,
        gamma=0.9,
        c=0.1,
    )

    # Train agent
    for _ in range(10_000):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, env.get_actions(state))
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

    # Test agent
    import pygame

    state = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.get_action(state, env.get_actions(state))
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    pygame.quit()


if __name__ == "__main__":
    main()
