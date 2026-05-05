import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "A2" / "app.py"), run_name="__main__")
