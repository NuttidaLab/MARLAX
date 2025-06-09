"""
GridWorld environments for MARLAX.

Defines a configurable multi-agent grid world with reward targets, penalties,
travel costs, and various reward regimes (r0–r4).
"""
from marlax.abstracts import Environment
from itertools import product
import random

class GridWorld(Environment):
    """
    Multi-agent grid environment with dynamic reward activation and penalties.

    Attributes:
        grid (tuple[int, int]): Dimensions (width, height) of the grid.
        agents (list[Agent]): Agent instances operating in the grid.
        target_rewards (list[float]): Reward for each agent upon correct target.
        together_reward (float): Bonus for agents co-located.
        travel_reward (float): Penalty (cost) for each move.
        wrong_zone_penalty (float): Penalty for entering a wrong reward zone.
        mismatch_penalty (float): Penalty if agents split between correct zones.
        possibilities (list[str]): Possible reward configurations.
        center_pos (tuple[int,int]): Coordinates of grid center.
        reward_place_to_coord (dict[str, tuple[tuple[int,int], ...]]): Maps target IDs to coordinates.
        moves (dict[str, tuple[int,int]]): Maps action names to (dx,dy) offsets.
        poss_act_combinations (list[tuple[str,...]]): All action combos per agent.
        steps_without_reward (int): Counter for steps without reward.
        no_reward_threshold (int): Steps limit before forced reset.
    """
    
    def __init__(self, grid, agents, target_rewards, together_reward, travel_reward, wrong_zone_penalty = -500, mismatch_penalty = -250):
        """
        Initialize the grid world parameters and agents.

        Args:
            grid (tuple[int,int]): Grid dimensions as (width, height).
            agents (list[Agent]): Agent instances present in the environment.
            target_rewards (list[float]): Reward values per agent for correct target.
            together_reward (float): Bonus reward if all agents share a cell.
            travel_reward (float): Cost (negative reward) per move.
            wrong_zone_penalty (float, optional): Penalty for entering a wrong zone. Defaults to -500.
            mismatch_penalty (float, optional): Penalty if agents split between two target zones. Defaults to -250.
        """
        self.grid = grid
        self.agents = agents # List of Agent instances.
        self.target_rewards = target_rewards
        self.together_reward = together_reward
        self.travel_reward = travel_reward
        self.wrong_zone_penalty = wrong_zone_penalty
        self.mismatch_penalty = mismatch_penalty
        
        # Active reward target managed by the environment.
        # It will be a tuple (like ('lr')) or None if not active.
        self.active_reward_target = None
        self.possibilities = []
        
        # Center of the grid.
        self.center_pos = (grid[0] // 2, grid[1] // 2)
        
        # Mapping from reward identifiers to board coordinates.
        self.reward_place_to_coord = {
            "u": ((grid[0] // 2, grid[1] - 1),),
            "r": ((grid[0] - 1, grid[1] // 2),),
            "d": ((grid[0] // 2, 0),),
            "l": ((0, grid[1] // 2),),
            "ur": ((grid[0] // 2, grid[1] - 1), (grid[0] - 1, grid[1] // 2)),
            "rd": ((grid[0] - 1, grid[1] // 2), (grid[0] // 2, 0)),
            "dl": ((grid[0] // 2, 0), (0, grid[1] // 2)),
            "ul": ((0, grid[1] // 2), (grid[0] // 2, grid[1] - 1)),
            "ud": ((grid[0] // 2, grid[1] - 1), (grid[0] // 2, 0)),
            "rl": ((grid[0] - 1, grid[1] // 2), (0, grid[1] // 2)),
        }
        
        self.moves = {
            'stay': (0, 0),
            'up':    (0, -1),
            'down':  (0, 1),
            'left':  (-1, 0),
            'right': (1, 0)
        }
        
        self.poss_act_combinations = list(product(self.moves.keys(), repeat=len(self.agents)))
        
        # For resetting when no rewards are collected over time.
        self.steps_without_reward = 0
        self.no_reward_threshold = 50

    def get_state(self):
        """
        Get the current global state representation.

        Returns:
            tuple: ((agent_positions), active_reward_target)
        """
        agent_positions = tuple(agent.position for agent in self.agents)
        # Use None if no active reward target.
        return (agent_positions, self.active_reward_target)

    def reset(self):
        """
        Randomly reposition agents and clear active rewards.

        Sets each agent to a random cell and chooses a new true_reward_target.
        """
        for agent in self.agents:
            agent.position = (random.randint(0, self.grid[0]-1),
                              random.randint(0, self.grid[1]-1))
        self.active_reward_target = None
        self.true_reward_target = random.choice(self.possibilities)

    def move_agents(self, actions):
        """
        Update agent positions based on provided actions.

        Args:
            actions (list[str]): Actions for each agent.
        """

        for idx, action in enumerate(actions):
            agent = self.agents[idx]
            dx, dy = self.moves.get(action, (0, 0))
            new_x = max(0, min(self.grid[0] - 1, agent.position[0] + dx))
            new_y = max(0, min(self.grid[1] - 1, agent.position[1] + dy))
            agent.position = (new_x, new_y)

    def get_possible_states(self):
        """
        Compute all possible next global states from current positions.

        Returns:
            list[tuple]: List of (positions, active_reward_target) for each action combo.
        """
        possible_positions = []
        for action_comb in self.poss_act_combinations:
            new_agent_positions = []
            for idx, action in enumerate(action_comb):
                agent = self.agents[idx]
                dx, dy = self.moves.get(action, (0, 0))
                new_x = max(0, min(self.grid[0] - 1, agent.position[0] + dx))
                new_y = max(0, min(self.grid[1] - 1, agent.position[1] + dy))
                new_agent_positions.append((new_x, new_y))
            possible_positions.append((tuple(new_agent_positions), self.active_reward_target))
        return possible_positions
        
    def step(self, actions):
        """
        Execute one time step in the environment:
          1. Move agents according to their actions.
          2. Check for reward activation (e.g., an agent reaching the center).
          3. Check if agents are at the designated reward positions.
          4. Apply travel penalty and together bonus.
          5. Reset if no reward is collected for too long.
        

        Args:
            actions (list[str]): Action for each agent.

        Returns:
            tuple: (next_state, rewards, info)
                next_state (tuple): New global state.
                rewards (list[float]): Reward per agent.
                info (dict): Diagnostics including activation and termination flags.
        """
        # Trail termination tracker
        terminated = False
        
        # 1. Move agents.
        self.move_agents(actions)
        # Update global state.
        rewards = [0 for _ in self.agents]
        
        # 2. Check for reward activation if none is active.
        activated = self.check_and_activate_rewards()
        
        # 3. Compute rewards based on agent positions and active reward target.
        collected, rewards = self.compute_rewards(rewards)
        
        reached_wrong_zone = self.check_wrong_reward_zones()
        
        went_to_mismatched = self.check_mismatch()
        
        # 4. Add together bonus if all agents are at the same position.
        if len(set(agent.position for agent in self.agents)) == 1:
            rewards = [r + self.together_reward for r in rewards]
        
        # 5. Add travel (energy loss) penalty.
        rewards = [r + self.travel_reward for r in rewards]
        
        if reached_wrong_zone:
            rewards = [r + self.wrong_zone_penalty for r in rewards]
        
        if went_to_mismatched:
            rewards = [r + self.mismatch_penalty for r in rewards]
        
        # Update no-reward counter.
        if collected\
        or (self.steps_without_reward > self.no_reward_threshold)\
        or reached_wrong_zone:
            # either a reward was collected
            # or moved too much
            # or someone reached a wrong zone
            terminated = True
        else:
            self.steps_without_reward += 1
        
        # Return the new global state as observation for all agents.
        next_state = self.get_state()
                
        info = {
            "activated": activated,
            "collected": collected,
            "terminated": terminated,
            "steps_without_reward": self.steps_without_reward,
        }
        
        # Reset should happen very last
        if terminated:
            self.steps_without_reward = 0
            self.reset()
        
        return next_state, rewards, info

    def check_and_activate_rewards(self):
        """
        Check if any agent is at the center and no reward target is active.
        If so, activate the reward target.
        This method is meant to be overridden by regime-specific environments.
        
        Returns:
            bool: True if activation occurred.
        """
        if self.active_reward_target is None:
            for agent in self.agents:
                if agent.position == self.center_pos:
                    self.active_reward_target = self.true_reward_target
                    return True
        return False

    def compute_rewards(self, rewards):
        """
        Compute rewards based on agent positions and active reward target.
        Modify the rewards list in place.
        
        Args:
            rewards (list[float]): Current rewards (modified in place).

        Returns:
            tuple: (collected, rewards_list)
                collected (bool): True if any reward was collected.
        """
        collected = False
        if self.active_reward_target:
            coords = self.reward_place_to_coord.get(self.active_reward_target, None)
            for reward_coord in coords:
                if all(agent.position == reward_coord for agent in self.agents):
                    for i in range(len(rewards)):
                        rewards[i] += self.target_rewards[i]
                    collected = True
                    break
        return collected, rewards
    
    def check_mismatch(self):
        """
        Detect if agents split between two correct target zones.

        Returns:
            bool: True if agents occupy both target cells.
        """
        if self.active_reward_target:
            coords = self.reward_place_to_coord.get(self.active_reward_target, None)
            if any(agent.position == coords[0] for agent in self.agents) and any(agent.position == coords[1] for agent in self.agents):
                return True
        return False
    
    def check_wrong_reward_zones(self):
        """
        Check if any agent enters a non-target reward zone.

        Returns:
            bool: True if a wrong-zone entry occurred.
        """
        if self.active_reward_target is not None:
            wrong_zones = {'u','r','d','l'} - set(self.active_reward_target)
            for zone in wrong_zones:
                coords = self.reward_place_to_coord.get(zone, None)
                for reward_coord in coords:
                    if any(agent.position == reward_coord for agent in self.agents):
                        return True
        return False
    
class GridWorld_r0(GridWorld):
    """
    Regime 0: Single-step reward at center regardless of target labels.
    """
    def __init__(self, grid, n_agents, target_rewards, together_reward, travel_reward):
        """
        Initialize regime-0 grid world (center-only reward).
        """
        super().__init__(grid, n_agents, target_rewards, together_reward, travel_reward)
        # Fixed reward target for regime 0.
        self.possibilities = [None]
        
    def compute_rewards(self, rewards):
        """
        Reward when any agent reaches the center cell.

        Returns:
            tuple: (collected, rewards)
        """
        collected = False
        if any(agent.position == self.center_pos for agent in self.agents):
            for i in range(len(rewards)):
                rewards[i] += self.target_rewards[i]
            collected = True
        return collected, rewards

class GridWorld_r1(GridWorld):
    """
    Regime 1: Single target 'rl' activated after center.
    """
    def __init__(self, grid, n_agents, target_rewards, together_reward, travel_reward):
        super().__init__(grid, n_agents, target_rewards, together_reward, travel_reward)
        self.possibilities = ["rl"]

class GridWorld_r2(GridWorld):
    """
    Regime 2: Single target 'ud' activated after center.
    """
    def __init__(self, grid, n_agents, target_rewards, together_reward, travel_reward):
        super().__init__(grid, n_agents, target_rewards, together_reward, travel_reward)
        self.possibilities = ["ud"]

class GridWorld_r3(GridWorld):
    """
    Regime 3: Multiple directional targets after center.
    """
    def __init__(self, grid, n_agents, target_rewards, together_reward, travel_reward):
        super().__init__(grid, n_agents, target_rewards, together_reward, travel_reward)
        self.possibilities = [
            "ur",
            "rd",
            "dl",
            "ul",
            "rl",
            "ud"
        ]
        
class GridWorld_r4(GridWorld):
    """
    Regime 4: Same as r3 but only returns current state.
    """
    def __init__(self, grid, n_agents, target_rewards, together_reward, travel_reward):
        super().__init__(grid, n_agents, target_rewards, together_reward, travel_reward)
        self.possibilities = [            
            "ur",
            "rd",
            "dl",
            "ul",
            "rl",
            "ud"
        ]
    
    def get_possible_states(self):
        """
        Override to return only current state (no branch expansion).

        Returns:
            list[tuple]: Single-element list containing current state.
        """
        return [self.get_state()]
        