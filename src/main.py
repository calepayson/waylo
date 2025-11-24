import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Runs waylo"""
    project_root = Path(__file__).parent.parent
    voc_data_path = project_root / "voc_data"

    # Check if voc_data dir exists, download if not
    if not voc_data_path.exists():
        logger.info("voc_data not found, running get_data.py")
        get_data_script = project_root / "scripts" / "get_data.py"
        subprocess.run(
            [sys.executable, str(get_data_script)], cwd=project_root, check=True
        )
    else:
        logger.info("voc_data found")

    logger.info("Starting waylo...")


if __name__ == "__main__":
    main()
