from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
# from fast_env_hiv import FastHIVPatient as HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


# environment without domain randomization
env_default = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)
# environment with domain randomization
env_dr = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)


class DQN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int]=[256, 256, 256, 256, 256],
    ) -> None:
        """Network architecture for the DQN agent.

        Args:
            input_dim (int): Dimension of the input.
            output_dim (int): Dimension of the output.
            hidden_dims (list[int], optional): Dimensions of the hidden layers.
                Defaults to [256, 256, 256, 256, 256].
        """
        super(DQN, self).__init__()
        
        self.net = nn.ModuleList()
        # input layer
        self.net.append(nn.Linear(input_dim, hidden_dims[0]))
        self.net.append(nn.ReLU())
        # hidden layers
        for i in range(len(hidden_dims) - 1):
            self.net.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.net.append(nn.ReLU())
        # output layer
        self.net.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*self.net)
        
        # weights initialization
        self.net.apply(self._weight_init)
    
    def _weight_init(self, layer: nn.Module) -> None:
        """Initialize the weights of the network using Xavier initialization.

        Args:
            layer (nn.Module): Layer of the network.
        """ 
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        batch_size: int,
        device: torch.device="cpu"
    ) -> None:
        """Replay buffer for storing the transitions.

        Args:
            capacity (int): Number of transitions to store.
            batch_size (int): Size of the mini-batch.
            device (torch.device, optional): Device to store the tensors on
                (e.g., "cpu" or "cuda"). Defaults to "cpu".
        """
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device
        
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer.

        Args:
            state (np.ndarray): State of the environment.
            action (int): Action taken in the state.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Next state of the environment.
            done (bool): Flag indicating the end of the episode.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a mini-batch of transitions from the replay buffer.

        Returns:
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]: Mini-batch of transitions
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(actions), dtype=torch.int64).to(self.device),
            torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device),
            torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        )

    def __len__(self) -> int:
        """Return the number of transitions stored in the replay buffer.

        Returns:
            int: Number of transitions stored in the replay buffer.
        """
        return len(self.buffer)


class ProjectAgent:
    
    # hyperparameters
    GAMMA = 0.98
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    REPLAY_BUFFER_SIZE = int(1e5)
    EPSILON_MAX = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.99
    TAU = 1.0
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 4,
        device: torch.device = "cpu",
    ) -> None:
        """Initialize the agent.

        Args:
            state_dim (int, optional): Dimension of the state space.
                Defaults to 6.
            action_dim (int, optional): Dimension of the action space.
                Defaults to 4.
            device (torch.device, optional): Device to store the tensors on.
                Defaults to "cpu".
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # online and target networks
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.policy_net.train()
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer and replay buffer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.LEARNING_RATE,
        )
        self.replay_buffer = ReplayBuffer(
            self.REPLAY_BUFFER_SIZE,
            self.BATCH_SIZE,
            device,
        )
        self.epsilon = self.EPSILON_MAX
        
    def train(self) -> None:
        """Single parameter update of the agent.
        """
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        # sample a mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # compute the target Q-values and the current Q-values
        max_next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = torch.addcmul(
            rewards,
            1 - dones,
            max_next_q_values,
            value=self.GAMMA,
        )
        q_values = self.policy_net(states).gather(1, actions.to(torch.long).unsqueeze(1))
        
        # optimize the model
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
    
    def epsilon_decay(self) -> None:
        """Decay the epsilon value.
        """
        self.epsilon = max(self.EPSILON_MIN, self.epsilon * self.EPSILON_DECAY)
    
    def update_target_network(self) -> None:
        """Update the target network using the policy network (exponential moving
        average).
        """
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] =\
                self.TAU * policy_state_dict[key]\
                    + (1 - self.TAU) * target_state_dict[key]
        self.target_net.load_state_dict(target_state_dict)
    
    def act(self, observation: np.ndarray, use_random : bool = False) -> int:
        """Select an action using a greedy or epsilon-greedy policy.

        Args:
            observation (np.ndarray): Observation from the environment.
            use_random (bool, optional): Flag to use epsilon-greedy policy.
                Defaults to False.

        Returns:
            int: Selected action.
        """
        # epsilon-greedy policy
        if use_random and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        # greedy policy
        with torch.no_grad():
            observation = torch.tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values = self.policy_net(observation)
            return int(torch.argmax(q_values))

    def save(self, path="model.pt") -> None:
        """Save the model parameters.

        Args:
            path (str, optional): Path to save the model parameters.
                Defaults to "model.pt".
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="model.pt") -> None:
        """Load the model parameters.

        Args:
            path (str, optional): Path to load the model parameters.
                Defaults to "model.pt".
        """
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()


if __name__ == "__main__":
    
    # hyperparameters
    MAX_EPISODES = 500
    TARGET_UPDATE_FREQUENCY = 2  # episodes between target network updates
    GRADIENT_STEPS = 3
    DR_EPISODES = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = ProjectAgent(
        env_default.observation_space.shape[0],
        env_default.action_space.n,
        device,
    )
    
    best_validation_score = 0
    best_validation_score_dr = 0
    
    total_rewards = []

    for episode in range(MAX_EPISODES):
        
        # switch between environments
        if episode % DR_EPISODES == 0:
            env = env_dr
        else:
            env = env_default
        
        # initialize the environment
        state, _ = env.reset()

        stop = False
        episode_rewards = []
        
        while not stop:
            # select an action (epsilon-greedy)
            action = agent.act(state, use_random=True)
            # perform the action
            next_state, reward, done, truncated, _ = env.step(action)
            episode_rewards.append(reward)
            
            # store the transition in the replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state

            # train the agent
            for _ in range(GRADIENT_STEPS):
                agent.train()
            
            stop = done or truncated
        
        agent.epsilon_decay()

        # update the target network periodically
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()
        
        # validation
        if episode > 100:
            validation_score = evaluate_HIV(agent=agent, nb_episode=5)
            print(f"Validation score: {validation_score:.2e}")
            validation_score_dr = evaluate_HIV_population(agent=agent, nb_episode=5)
            print(f"Validation score DR: {validation_score_dr:.2e}")
            
            # Pareto optimality
            if validation_score > best_validation_score:
                print("Saving the best model")
                best_validation_score = validation_score
                agent.save(f"model_{episode}.pt")
            elif validation_score_dr > best_validation_score_dr:
                print("Saving the best model DR")
                best_validation_score_dr = validation_score_dr
                agent.save(f"model_{episode}.pt")

        total_reward = np.sum(episode_rewards)
        print(f"Episode {episode + 1}/{MAX_EPISODES}, Total Reward: {total_reward:.2f}")
        print("Epsilon: ", agent.epsilon)
        print("Replay buffer size: ", len(agent.replay_buffer))
        print()
        total_rewards.append(total_reward)

    env_default.close()
    env_dr.close()
    
    # Save the rewards
    # np.save("rewards.npy", total_rewards)
