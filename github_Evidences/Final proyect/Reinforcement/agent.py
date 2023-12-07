import numpy as np

class FastestRouteFindingRobotEnv:
    def __init__(self, map_size):
        self.map_size = map_size
        self.robot_position = 0
        self.target_position = map_size - 1
        self.observation_space = np.arange(map_size)
        self.action_space = np.array([0, 1])  

    def reset(self):
        self.robot_position = 0
        return self.robot_position

    def step(self, action):
        if action == 0:  # Move right
            self.robot_position = min(self.robot_position + 1, self.map_size - 1)
        elif action == 1:  # Move left
            self.robot_position = max(self.robot_position - 1, 0)

        done = self.robot_position == self.target_position

        # Penalize the distance to the target
        distance_penalty = -0.1 * abs(self.robot_position - self.target_position)

        # Reward for reaching the target
        reward = 1 if done else 0

        reward += distance_penalty

        return self.robot_position, reward, done, {}

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.01, discount_factor=0.95, exploration_prob=0.5, exploration_decay=0.001):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.action_space_size) 
        else:
            return np.argmax(self.q_table[state, :])  

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[state, action] += self.learning_rate * (
                reward + self.discount_factor * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
        )

    def decay_exploration_prob(self):
        self.exploration_prob = max(0.1, self.exploration_prob - self.exploration_decay)

env = FastestRouteFindingRobotEnv(map_size=15)
agent = QLearningAgent(state_space_size=env.observation_space.size, action_space_size=env.action_space.size)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update_q_table(state, action, reward, next_state)

        state = next_state

    agent.decay_exploration_prob()

    if episode % 100 == 0:
        print(f"Episode {episode}, Exploration Probability: {agent.exploration_prob}, Steps: {env.robot_position}, Reward: {reward}")
