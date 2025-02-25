import os
import logging
from datetime import datetime


# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Define your reporting folder (ensure this exists)
REPORTS_FOLDER = "QuantStats_Reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Constants
COLAB = False

if os.getenv("COLAB_RELEASE_TAG"):
    COLAB = True
    
    import gc
    from google.colab import drive
    
    sys.path.append('/content/drive/My Drive/AFP/Code/Download_This_Folder')  # Update this path to your folder
    drive.mount('/content/drive')

DB_PATH = r"1_financial_data_long.db"
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 12, 31)