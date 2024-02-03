import numpy as np

class HAL_Agent():
    def __init__(self, ppo_agent=None, env=None):
        if ppo_agent is not None:
            print("Loading PPO Agent")
        else:
            print("Loading IK Agent")
        self.ppo_agent = ppo_agent
        self.env = env

    def select_action(self, obs):
        action = None
        if self.ppo_agent is not None:
            action, _ = self.ppo_agent.predict(obs, deterministic=True)
        else:
            dist_list = np.zeros(10)
            for j in range(10):
                delta_state = self.env.simulate_step(j)
                dist_list[j] = np.linalg.norm(delta_state)
            action = np.argmin(dist_list)
            _, _, _, _ = self.env.step(action)

        return action

