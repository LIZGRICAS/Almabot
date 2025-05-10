import logging
from datetime import datetime
import os

# Define log directory and file path
log_dir = '/app/src/api/logs'
log_file = os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log')

# Ensure log directory exists and has proper permissions
os.makedirs(log_dir, exist_ok=True)
os.chmod(log_dir, 0o777)

# Create file handler with proper permissions
file_handler = logging.FileHandler(log_file, mode='w')  # Use write mode to create new file
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)

# Create logger for the API
logger = logging.getLogger('api_logger')

# Test logging
logger.info("Logging system initialized")
