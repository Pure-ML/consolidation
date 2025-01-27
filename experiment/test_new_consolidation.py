import pandas as pd
import logging
from pathlib import Path
from NewConsolidation import DatasetAgnosticConsolidator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_csv_path() -> Path:
    """Prompt user for CSV file path and validate it."""
    while True:
        file_path = input("\nEnter the path to your CSV file: ").strip()
        path = Path(file_path)
        
        if not file_path:
            print("Please enter a file path.")
            continue
            
        if not path.exists():
            print(f"File not found: {file_path}")
            continue
            
        if path.suffix.lower() != '.csv':
            print(f"File must be a CSV file: {file_path}")
            continue
            
        return path

def select_column(df: pd.DataFrame) -> str:
    """Let user select a column from the DataFrame."""
    while True:
        print("\nAvailable columns:")
        for idx, col in enumerate(df.columns, 1):
            print(f"{idx}. {col}")
        
        try:
            choice = input("\nEnter the number of the column to consolidate: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(df.columns):
                return df.columns[idx]
            else:
                print("Invalid selection. Please choose a number from the list.")
        except ValueError:
            print("Please enter a valid number.")

def get_parameters() -> dict:
    """Get optional parameters from user."""
    params = {}
    
    print("\nEnter parameters (press Enter to use defaults):")
    
    try:
        min_cluster = input("Minimum cluster size [5]: ").strip()
        params['min_cluster_size'] = int(min_cluster) if min_cluster else 5
        
        similarity = input("Similarity threshold [0.85]: ").strip()
        params['similarity_threshold'] = float(similarity) if similarity else 0.85
        
        confidence = input("Confidence threshold [0.7]: ").strip()
        params['confidence_threshold'] = float(confidence) if confidence else 0.7
        
        batch_size = input("Batch size for embeddings [20]: ").strip()
        params['batch_size'] = int(batch_size) if batch_size else 20
        
    except ValueError as e:
        print(f"Invalid input: {e}. Using default values.")
        return {
            'min_cluster_size': 5,
            'similarity_threshold': 0.85,
            'confidence_threshold': 0.7,
            'batch_size': 20
        }
    
    return params

def confirm_action(prompt: str) -> bool:
    """Ask user for confirmation."""
    while True:
        response = input(f"\n{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")

def main():
    print("=== Dataset Agnostic Consolidation ===")
    
    try:
        # Get CSV file
        input_path = get_csv_path()
        
        # Load the dataset
        logger.info(f"Loading dataset from {input_path}")
        df = pd.read_csv(input_path)
        
        # Select column
        column = select_column(df)
        logger.info(f"Selected column: {column}")
        
        # Get parameters
        params = get_parameters()
        
        # Initialize consolidator
        consolidator = DatasetAgnosticConsolidator(**params)
        
        # Preview groups first
        logger.info(f"Analyzing column: {column}")
        _, preview_metrics = consolidator.consolidate(
            df=df,
            column=column,
            preview_only=True
        )
        
        # Ask for confirmation
        if not confirm_action("Would you like to apply these consolidations?"):
            print("\nOperation cancelled. No changes were made.")
            return
        
        # Set output path
        output_path = input_path.parent / f"{input_path.stem}_consolidated{input_path.suffix}"
        
        # Run consolidation
        logger.info(f"Applying consolidation to column: {column}")
        df_consolidated, metrics = consolidator.consolidate(
            df=df,
            column=column
        )
        
        # Save results
        df_consolidated.to_csv(output_path, index=False)
        logger.info(f"Saved consolidated dataset to {output_path}")
        
        # Print metrics
        logger.info("\nConsolidation Metrics:")
        logger.info(f"Original unique values: {metrics['original_unique_values']}")
        logger.info(f"Consolidated unique values: {metrics['consolidated_unique_values']}")
        logger.info(f"Groups formed: {metrics['groups_formed']}")
        logger.info(f"Token usage: {metrics['token_usage']}")
        logger.info(f"Runtime: {metrics['runtime_seconds']:.2f} seconds")
        
        # Calculate reduction percentage
        reduction = (
            (metrics['original_unique_values'] - metrics['consolidated_unique_values']) 
            / metrics['original_unique_values'] 
            * 100
        )
        logger.info(f"Reduction in unique values: {reduction:.1f}%")
        
    except Exception as e:
        logger.error(f"Error during consolidation: {str(e)}")
        raise
    
    print("\nDone! Press Enter to exit...")
    input()

if __name__ == "__main__":
    main()
