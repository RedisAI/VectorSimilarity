import pytest
import os
import logging
import sys
from datetime import datetime


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
    per_test_logs = request.config.getoption("--per-test-logs")
    log_prefix = request.config.getoption("--log-file-prefix")
    log_dir = getattr(request.config, '_log_dir', 'logs/tests/flow')
    
    if per_test_logs:
        # Create per-test log file
        test_name = request.node.name
        test_file = request.node.parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        log_filename = os.path.join(log_dir, f"{test_file}_{test_name}_{timestamp}.log")
    else:
        # Use single log file (compatible with Makefile --log-file option)
        log_filename = os.path.join(log_dir, f"{log_prefix}.log")
    
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
    
    # Store log info for VecSim integration
    request.node.log_filename = log_filename
    request.node.test_logger = logger
    
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


@pytest.fixture(autouse=True)
def setup_vecsim_logging(request):
    """Set up VecSim C++ logging integration."""
    if hasattr(request.node, 'log_filename'):
        log_filename = request.node.log_filename
        
        # Try to set up VecSim logging callback
        def vecsim_log_callback(ctx, level, message):
            """Callback function for VecSim logging."""
            try:
                with open(log_filename, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] [{level}] {message}\n")
                    f.flush()
            except Exception as e:
                # Fallback to stderr if file logging fails
                print(f"VecSim logging error: {e}", file=sys.stderr)
        
        # Try different methods to set VecSim logging
        vecsim_logging_set = False
        
        try:
            # Method 1: Try direct function import
            from VecSim import VecSim_SetLogCallbackFunction
            VecSim_SetLogCallbackFunction(vecsim_log_callback)
            vecsim_logging_set = True
            if hasattr(request.node, 'test_logger'):
                request.node.test_logger.debug("VecSim logging callback set via VecSim_SetLogCallbackFunction")
        except (ImportError, AttributeError):
            pass
        
        if not vecsim_logging_set:
            try:
                # Method 2: Try alternative function name
                from VecSim import set_log_callback
                set_log_callback(vecsim_log_callback)
                vecsim_logging_set = True
                if hasattr(request.node, 'test_logger'):
                    request.node.test_logger.debug("VecSim logging callback set via set_log_callback")
            except (ImportError, AttributeError):
                pass
        
        if not vecsim_logging_set:
            try:
                # Method 3: Try ctypes approach for direct C++ integration
                import VecSim
                import ctypes
                
                # Get the shared library handle
                lib_path = VecSim.__file__
                lib = ctypes.CDLL(lib_path)
                
                # Define the callback function type
                LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
                
                def c_vecsim_callback(ctx, level, message):
                    level_str = level.decode('utf-8') if level else 'INFO'
                    message_str = message.decode('utf-8') if message else ''
                    vecsim_log_callback(ctx, level_str, message_str)
                
                # Create the callback
                callback = LOG_CALLBACK(c_vecsim_callback)
                
                # Try to call the C function
                try:
                    lib.VecSim_SetLogCallbackFunction(callback)
                    vecsim_logging_set = True
                    if hasattr(request.node, 'test_logger'):
                        request.node.test_logger.debug("VecSim logging callback set via ctypes")
                except AttributeError:
                    pass
                    
            except Exception:
                pass
        
        if not vecsim_logging_set and hasattr(request.node, 'test_logger'):
            request.node.test_logger.warning("Could not set VecSim logging callback - C++ logs will not be captured")
    
    yield
    
    # Reset VecSim logging after test
    try:
        from VecSim import VecSim_SetLogCallbackFunction
        VecSim_SetLogCallbackFunction(None)
    except (ImportError, AttributeError):
        try:
            from VecSim import set_log_callback
            set_log_callback(None)
        except (ImportError, AttributeError):
            pass
