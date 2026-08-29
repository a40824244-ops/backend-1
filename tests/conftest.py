import os
import sys

# Ensure current project root directory is added to sys.path for test runners
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
