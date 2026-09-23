import yaml
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> Optional[Dict[str, Any]]:
    """Loading YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error while loading config: {e}")
        return None

def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Loading dataset based on config path."""
    path = Path(config['data']['raw_path'])
    index_col = config['data'].get('index_column')
    
    if path.exists():
        if path.suffix == '.csv':
            df = pd.read_csv(path)
            df = pd.read_csv(path, index_col=index_col)
        elif path.suffix in ['.xls', '.xlsx']:
            df = pd.read_excel(path)
            df = pd.read_excel(path, index_col=index_col)
        else:
            raise ValueError("File format not supported (use CSV or Excel).")
        logger.info(f"Data loaded successfully: {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    raise FileNotFoundError(f"File was not found in: {path}")

def drop_unwanted_columns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Remove columns specified in the config file from the dataframe."""
    cols_to_drop = config['data'].get('drop_columns', [])
    # Checking if columns exist before dropping to avoid errors
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"Dropped columns: {existing_cols}")
    return df