import shutil
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pickle

class Tracer:
    
    def __init__(self, log_path):
        """
        Initialize the tracer.
        
        Args:
            log_path (str): Path to the log directory.
        """
        
        # Logger attributes
        self.logger_active = False
        self.log_buffer = []
        self.flush_every = None
        self.regime_idx = None
        self.log_filename = None
        self.parquet_writer = None
        self.log_path = log_path
        
        # Remove folder if it exists
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
    
    def _init_logger(self, flush_every, regime_idx, who="training"):
        """
        Initialize the logger that appends rows to an HDF5 file.
        
        Args:
            flush_every (int): Number of frames to buffer before writing to disk.
            regime_idx (int): Regime identifier that is recorded with each row.
        """
        # Ensure the log path exists.
        os.makedirs(self.log_path+"/logs", exist_ok=True)
        # Build the full log filename using the provided log_path.
        self.log_filename = os.path.join(self.log_path+"/logs", f"{who}_{regime_idx}.h5")
        self.flush_every = flush_every
        self.regime_idx = regime_idx
        self.log_buffer = []
        self.log_store = pd.HDFStore(self.log_filename, mode='a')
        self.logger_active = True
        
    def _init_logger(self, flush_every, regime_idx, who="training"):
        """
        Initialize the logger that appends rows to a single Parquet file.
        
        Args:
            flush_every (int): Number of frames to buffer before writing to disk.
            regime_idx (int): Regime identifier that is recorded with each row.
            who (str): Label to differentiate logs (e.g. 'training' or 'test').
        """
        # Ensure the log directory exists.
        os.makedirs(self.log_path+"/logs", exist_ok=True)
        # Create the log filename.
        self.log_filename = os.path.join(self.log_path+"/logs", f"{who}_{regime_idx}.parquet")
        self.flush_every = flush_every
        self.regime_idx = regime_idx
        self.log_buffer = []
        self.parquet_writer = None  # We'll initialize this on the first flush
        self.logger_active = True
    
    def _flush_buffer(self):
        """Flush the log buffer to the Parquet file using PyArrow."""
        if not self.log_buffer:
            return
        
        # Convert the buffer to a pandas DataFrame.
        df = pd.DataFrame(self.log_buffer)
        # Convert the DataFrame to a PyArrow Table.
        table = pa.Table.from_pandas(df)
        
        if self.parquet_writer is None:
            # Create a ParquetWriter with the schema inferred from the first flush.
            self.parquet_writer = pq.ParquetWriter(self.log_filename, table.schema)
        
        self.parquet_writer.write_table(table)
        self.log_buffer = []
    
    def _log_frame(self, step, next_state, rewards, info):
        """
        Build a row from the current frame data and add it to the log buffer.
        Flush to disk when the buffer size reaches flush_every.
        """
        # Expect next_state to be a tuple: (agent_states, reward_loc)
        agent_states, reward_loc = next_state
        row = {
            "regime_idx": self.regime_idx,
            "frame_idx": step,
            "reward_loc": reward_loc,
            "activated": info["activated"],
            "collected": info["collected"],
            "terminated": info["terminated"],
            "steps_without_reward": info["steps_without_reward"]
        }
        # Add agent coordinates.
        for i, (x, y) in enumerate(agent_states):
            row[f"a{i+1}x"] = x
            row[f"a{i+1}y"] = y
        # Add rewards.
        for i, reward in enumerate(rewards):
            row[f"r{i+1}"] = reward
        
        self.log_buffer.append(row)
        if len(self.log_buffer) >= self.flush_every:
            self._flush_buffer()
    
    def _flush_logger(self):
        """
        Flush any remaining buffered rows and close the Parquet writer.
        """
        if self.logger_active:
            if self.log_buffer:
                self._flush_buffer()
            if self.parquet_writer is not None:
                self.parquet_writer.close()
            self.logger_active = False
            
    def export_agents(self, env):
        """
        Export agents to a file.
        """
        os.makedirs(self.log_path+"/qvals", exist_ok=True)
        for idx, agent in enumerate(env.agents):
            filename = f"{self.log_path}/qvals/agent_{idx}.pkl"
            with open(filename, "wb") as file:
                pickle.dump(agent.q_table, file)
    
    def import_agents(self, agent):
        """
        Import agents from a file.
        """
        all_agents = []
        
        # load all the pkl files in the folder
        for filename in os.listdir(self.log_path+"/qvals"):
            if filename.endswith(".pkl"):
                with open(f"{self.log_path}/qvals/{filename}", "rb") as file:
                    a = agent()
                    a.q_table = pickle.load(file)
                    all_agents.append(a)
        return all_agents