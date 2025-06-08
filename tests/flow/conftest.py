import pytest
import os
import logging
from datetime import datetime


def pytest_addoption(parser):
    parser.addoption("--log-dir", action="store", default="logs/tests/flow",
                    help="Directory to store log files")


@pytest.fixture(scope="session", autouse=True)
def setup_global_logging(request):
    """Session-level logging setup."""
    log_dir = request.config.getoption("--log-dir")
    
    if not os.path.isabs(log_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    request.config._log_dir = log_dir
    return log_dir


@pytest.fixture(autouse=True)
def setup_test_logging(request):
    """Per-test logging setup."""
    log_dir = getattr(request.config, '_log_dir', 'logs/tests/flow')
    
    test_name = request.node.name
    test_file = request.node.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_filename = os.path.join(log_dir, f"{test_file}_{test_name}_{timestamp}.log")

    logger = logging.getLogger(f"test_{test_name}")
    logger.setLevel(logging.DEBUG)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"=== Starting test: {test_name} ===")
    
    yield logger
    
    logger.info(f"=== Finished test: {test_name} ===")
    file_handler.close()
    logger.removeHandler(file_handler)
