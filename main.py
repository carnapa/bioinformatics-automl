from src.utils import load_config, load_data, drop_unwanted_columns
from src.eda import run_eda
from src.models import run_model_training
import logging

# Initialize logging configuration for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("--- Initialising bioinformatics pipeline ---")
    
    # Loading config
    config = load_config('config.yaml')
    if not config:
        return

  
    try:
        # Loading data
        df = load_data(config)
    
        # Previewing data
        logger.info("Data Preview (first 5 rows):")
        logger.info(f"\n{df.head()}")

        # Droping unwanted columns
        df = drop_unwanted_columns(df, config)
        
        # 3. Running EDA
        if config['eda']['generate_pca'] or config['output']['save_plots']:
            run_eda(df, config)
            
        logger.info("Pipeline execution finished.")
        
    except Exception as e:
        logger.error(f"Error occurred during execution: {e}")

if __name__ == "__main__":
    main()