from baselines.dlinear import Model as DLinear
from baselines.patch_linear import Model as PatchLinear
from baselines.lstm import Model as LSTM
from baselines.tcn import Model as TCN
from baselines.simple import Persistence, Linear
from baselines.dlinear_gauss import DLinearGauss

__all__ = ["DLinear", "DLinearGauss", "PatchLinear", "LSTM", "TCN", "Persistence", "Linear"]
