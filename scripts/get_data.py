import kagglehub
import shutil
from pathlib import Path

path = kagglehub.dataset_download("aladdinpersson/pascalvoc-yolo")
target = Path.cwd() / "data"

shutil.move(path, target)
print("Path to dataset files:", target)
