import pytest
import os
import logging
from datetime import datetime

def pytest_configure(config):
    """Set up initial logging before any tests run."""
    # Get log directory from command line or use default
    log_dir = config.getoption("--log-dir", default="logs/tests/flow")
    print(f"Using log directory========================================================================: {log_dir}")
    # Convert relative path to absolute
    if not os.path.isabs(log_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a global log file for initialization logs
    init_log_file = os.path.join(log_dir, "pytest_init.log")
    
    # Set up a basic file logger for initialization
    init_logger = logging.getLogger("pytest_init")
    init_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in init_logger.handlers[:]:
        init_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(init_log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    init_logger.addHandler(file_handler)
    
    # Set VecSim log context for C++ logging integration at initialization
    try:
        from VecSim import set_log_context
        init_logger.info("Setting initial VecSim log context")
        print("Setting initial VecSim log context")
        set_log_context("pytest_init", "flow")
        init_logger.info("VecSim initial logging context set successfully")
    except (ImportError, AttributeError):
        init_logger.error("Failed to set VecSim logging context, likely not imported")
    
    # Store log directory in config for other fixtures
    config._log_dir = log_dir
    
    init_logger.info("Pytest initialization complete")

@pytest.fixture(autouse=True)
def setup_test_logging(request):
    """Per-test logging setup that works with both Makefile and direct pytest calls."""
    
    # Get configuration
    log_dir = getattr(request.config, '_log_dir', 'logs/tests/flow')
    
    # Get test name for logging
    test_name = request.node.name
    test_file = request.node.parent.name

    # Create per-test log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
    log_filename = os.path.join(log_dir, f"{test_file}_{test_name}_{timestamp}.log")

    # Set VecSim log context for C++ logging integration - use the full filename
    try:
        from VecSim import set_log_context
        # Extract just the filename without path and extension for the C++ context
        log_basename = os.path.splitext(os.path.basename(log_filename))[0]
        print(f"Setting log context for VecSim: {log_basename}")
        set_log_context(log_basename, "flow")
        print("VecSim logging context set successfully.")
    except (ImportError, AttributeError):
        print("VecSim logging context not set, likely not imported.")

    # Set up Python logging for this test
    logger = logging.getLogger(f"test_{test_name}")
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    # Log test start
    logger.info(f"=== Starting test: {test_name} ===")
    logger.info(f"Log file: {log_filename}")

    yield logger

    # Log test end and cleanup
    logger.info(f"=== Finished test: {test_name} ===")
    file_handler.close()
    logger.removeHandler(file_handler)

@pytest.fixture
def test_logger(setup_test_logging):
    """Convenience fixture to get the test logger."""
    return setup_test_logging
