"""Double-click launcher: starts VocaVision with no console window."""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)
runpy.run_path(os.path.join(here, "vocavision_app.py"), run_name="__main__")
