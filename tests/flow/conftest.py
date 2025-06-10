import pytest
import os
import logging
import sys
from datetime import datetime


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # We only look at actual failing test calls, not setup/teardown
    if rep.when == "call" and rep.failed:
        # Get the test logger
        test_name = item.name
        logger = logging.getLogger(f"test_{test_name}")
        
        # Log the failure details
        logger.error(f"TEST FAILED: {rep.nodeid}")
        if hasattr(rep, "longrepr"):
            logger.error(f"Failure details:\n{rep.longreprtext}")


def pytest_addoption(parser):
    """Add command line options for logging configuration."""
    parser.addoption("--log-file-prefix", action="store", default="vecsim",
                    help="Prefix for log files")
    parser.addoption("--log-dir", action="store", default="logs/tests/flow",
                    help="Directory to store log files")
    parser.addoption("--per-test-logs", action="store_true", default=True,
                    help="Create separate log file for each test")


@pytest.fixture(scope="session", autouse=True)
def setup_global_logging(request):
    """Session-level logging setup."""
    log_dir = request.config.getoption("--log-dir")
    print(f"Log directory from command line: {log_dir}")
    # Convert relative path to absolute
    if not os.path.isabs(log_dir):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Store log directory in session for other fixtures
    request.config._log_dir = log_dir
    
    return log_dir


@pytest.fixture(autouse=True)
def setup_test_logging(request):
    """Per-test logging setup that works with both Makefile and direct pytest calls."""
    
    # Get configuration
    # per_test_logs = request.config.getoption("--per-test-logs")
    # log_prefix = request.config.getoption("--log-file-prefix")
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
    # else:
    #     # Use single log file (compatible with Makefile --log-file option)
    #     log_filename = os.path.join(log_dir, f"{log_prefix}.log")
    
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



