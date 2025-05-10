import logging
from datetime import datetime
import os

# Define log directory and file path
log_dir = '/app/src/api/logs'
log_file = os.path.join(log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log')

# Ensure log directory exists
os.makedirs(log_dir, exist_ok=True)

# Set permissions for log directory and file
os.chmod(log_dir, 0o777)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Use append mode
        logging.StreamHandler()
    ]
)

# Create logger for the API
logger = logging.getLogger('api_logger')
