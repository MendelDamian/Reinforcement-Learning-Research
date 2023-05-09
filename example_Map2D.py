from rl_tools.agent import QAgent
from rl_tools.env import Map2D


def main():
    # Create environment
    env = Map2D(
        shape=(10, 10),
        start=(0, 0),
        goal=(9, 9),
        iterations=100,
    )

    # Create agent
    agent = QAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
    )

    # Train agent
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, env.get_actions(state))
            next_state, reward, done = env.step(action)
            print(state, action, reward, next_state, done)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            # env.render()

    # Test agent
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = agent.get_best_action(state, env.get_actions(state))
        next_state, reward, done = env.step(action)
        state = next_state
        env.render()


if __name__ == "__main__":
    main()
