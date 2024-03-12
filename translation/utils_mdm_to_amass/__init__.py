import sys
import importlib

sys.modules["psbody"] = importlib.import_module("translation.utils_mdm_to_amass.mesh-master")
sys.modules["psbody.mesh"] = importlib.import_module("translation.utils_mdm_to_amass.mesh-master.mesh")
