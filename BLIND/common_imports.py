# common_imports.py
# Description: Central file for importing common external libraries.
# Project-internal imports (e.g., from models import ...) should NOT be here.

# Standard Python Libraries
import os
import sys
import json
import yaml
import argparse
import logging
import time
import random
from typing import Dict, List, Optional, Tuple, Any # Common typing imports

# Core Scientific Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# PyTorch Utilities
from torch.utils.data import DataLoader as TorchDataLoader # For non-PyG data handling

# PyTorch Geometric (PyG)
from torch_geometric.data import Data as PyGData # Renamed to avoid conflict
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import ChebConv
from torch_cluster import knn_graph # For graph construction

# Other Third-Party Libraries
from sklearn.model_selection import train_test_split
import h5py # For HDF5 file operations# common_imports.py
# Description: Central file for importing common external libraries.
# Project-internal imports (e.g., from models import ...) should NOT be here.

# Standard Python Libraries
import os
import sys
import json
import yaml
import argparse
import logging
import time
import random
from typing import Dict, List, Optional, Tuple, Any # Common typing imports

# Core Scientific Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# PyTorch Utilities
from torch.utils.data import DataLoader as TorchDataLoader # For non-PyG data handling

# PyTorch Geometric (PyG)
from torch_geometric.data import Data as PyGData # Renamed to avoid conflict
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import ChebConv
from torch_cluster import knn_graph # For graph construction

# Other Third-Party Libraries
from sklearn.model_selection import train_test_split
