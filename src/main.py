import logging
import subprocess
import sys
from pathlib import Path

from config import CurrentConfig as Conf


logging.basicConfig(level=Conf.log_level)
logger = logging.getLogger(__name__)

config = Conf()


def ensure_voc_data(project_root: Path) -> None:
    """Download VOC data if not present."""
    voc_data_path = project_root / "voc_data"
    if not voc_data_path.exists():
        logger.info("voc_data not found, running get_data.py")
        get_data_script = project_root / "scripts" / "get_data.py"
        subprocess.run(
            [sys.executable, str(get_data_script)], cwd=project_root, check=True
        )
    else:
        logger.info("voc_data found")


def main():
    """Runs waylo."""
    logger.debug("Starting main...")
    project_root = Path(__file__).parent.parent

    ensure_voc_data(project_root)

    logger.info("Starting waylo...")


if __name__ == "__main__":
    main()
