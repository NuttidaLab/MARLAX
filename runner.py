# %%
from marlax.agents import QAgent, QValueAgent
from marlax.envs import GridWorld_r0, GridWorld_r3, GridWorld_r4
from marlax import Engine, Tracer

# %%
from joblib import Parallel, delayed

# %%
def train_and_test( seed=42, 
                    n_agents=2,
                    grid_size=(11,11),
                    target_reward=100,
                    together_reward=0,
                    travel_reward=-1,
                    epsilon_start=0.99,
                    epsilon_end=0.4,
                    alpha=0.1,
                    gamma=0.9 ): 

    # Set the random seed for reproducibility.
    # random.seed(seed)

    # Agents
    target_rewards = [target_reward] * n_agents  # Reward for each agent when target is met
    agents = [QValueAgent() for _ in range(n_agents)] 
    # agents = [QAgent() for _ in range(n_agents)] 

    # List the environments and train sequentially.
    environments = [GridWorld_r0, GridWorld_r3]
    nsteps = [1_000_000, 200_000_000]
    
    tracer = Tracer(f"store/{seed}")
    trainer = Engine(epsilon_start, epsilon_end, epsilon_test=0.0)
    
    for (i, e), steps in zip(enumerate(environments), nsteps):
        # Create one environment per regime.
        environment = e(grid_size, agents, target_rewards, together_reward, travel_reward)
        trainer.train(environment, tracer, num_steps=steps, alpha=alpha, gamma=gamma, regime_idx=i)
        trainer.test(environment, tracer, num_steps=10_000_00, regime_idx=i)
    tracer.export_agents(environment)

# %%
if __name__ == '__main__':
    seeds = list(range(100))
    results = Parallel(n_jobs=-1)(delayed(train_and_test)(seed=s) for s in seeds)


