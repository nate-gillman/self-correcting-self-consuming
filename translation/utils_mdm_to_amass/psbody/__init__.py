import sys
import importlib

# not a great practice, but we must do this because we need to compile mesh ourselves, instead of through pip
sys.modules["psbody"] = importlib.import_module("translation.utils_mdm_to_amass.mesh-master")
sys.modules["psbody.mesh"] = importlib.import_module("translation.utils_mdm_to_amass.mesh-master.mesh") 
