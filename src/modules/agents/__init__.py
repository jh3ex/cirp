REGISTRY = {}

from .rnn_agent import RNNAgent
from .gcn_agent import TRANSAgent, GATAgent, TRANSPPOAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["transformer"] = TRANSAgent
REGISTRY["trans_ppo"] = TRANSPPOAgent
REGISTRY["gat"] = GATAgent
