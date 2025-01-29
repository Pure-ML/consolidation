import os
import pandas as pd
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from IntelligentConsolidation import ImprovedRAGAgent, ValueGroup

class TestBench:
    def __init__(self, 
                 datasets_dir: str = "datasets",
                 results_dir: str = "results",
                 max_unique_values: int = 1000):
        self.datasets_dir = datasets_dir
        self.results_dir = results_dir
        self.max_unique_values = max_unique_values
        
        # Create directories if they don't exist
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize agent
        self.agent = ImprovedRAGAgent(model="gpt-4")
    
    def _get_available_datasets(self) -> List[str]:
        """Get list of available CSV files in the datasets directory."""
        return [f for f in os.listdir(self.datasets_dir) if f.endswith('.csv')]
    
    def _sample_dataset(self, df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Sample the dataset if requested."""
        if sample_size and sample_size < len(df):
            return df.sample(n=sample_size, random_state=42)
        return df
    
    def _process_column(self, df: pd.DataFrame, column: str) -> Dict:
        """Process a single column and return results."""
        unique_values = df[column].dropna().unique()
        if len(unique_values) > self.max_unique_values:
            print(f"Warning: Column {column} has {len(unique_values)} unique values. Processing only first {self.max_unique_values}.")
            # Sample unique values if over limit
            unique_values = pd.Series(unique_values).sample(n=self.max_unique_values, random_state=42)
            df = df[df[column].isin(unique_values)]
        
        return {
            "column": column,
            "total_rows": len(df),
            "unique_values": len(unique_values),
            "sample_values": unique_values[:5].tolist()
        }
    
    def _save_results(self, dataset_name: str, results: Dict):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add dataset stats to filename
        total_rows = results["total_rows"]
        sampled_rows = results["sampled_rows"] or total_rows
        filename = f"{dataset_name.replace('.csv', '')}_{sampled_rows}rows_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Get cost metrics
        cost_metrics = self.agent.cost_tracker.get_costs()
        
        # Add dataset information and costs to results
        results["dataset_info"] = {
            "name": dataset_name,
            "total_rows": total_rows,
            "rows_processed": sampled_rows,
            "sampling_ratio": round(sampled_rows / total_rows * 100, 2) if total_rows > 0 else 100,
            "processing_timestamp": timestamp
        }
        
        results["cost_metrics"] = {
            "chat_input_tokens": cost_metrics["chat_input_tokens"],
            "chat_output_tokens": cost_metrics["chat_output_tokens"],
            "total_tokens": cost_metrics["chat_input_tokens"] + cost_metrics["chat_output_tokens"],
            "total_cost": cost_metrics["total_cost"]
        }
        
        # Convert ValueGroup objects to dictionaries
        for column_result in results["column_results"]:
            if "groups" in column_result:
                column_result["groups"] = [
                    {
                        "values": list(group.values),
                        "canonical_form": group.canonical_form,
                        "similarity_score": group.similarity_score
                    }
                    for group in column_result["groups"]
                ]
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {filepath}")
        
        # Print summary
        print("\nDataset Summary:")
        print(f"  • Total rows in dataset: {total_rows:,}")
        print(f"  • Rows processed: {sampled_rows:,}")
        print(f"  • Sampling ratio: {results['dataset_info']['sampling_ratio']}%")
        print("\nCost Summary:")
        print(f"  • Input tokens: {cost_metrics['chat_input_tokens']:,}")
        print(f"  • Output tokens: {cost_metrics['chat_output_tokens']:,}")
        print(f"  • Total cost: ${cost_metrics['total_cost']:.4f}")

    def _select_canonical_forms(self, groups: List[ValueGroup]) -> List[ValueGroup]:
        """Interactive selection of canonical forms for groups."""
        print(f"\nFound {len(groups)} groups of similar values (sorted by similarity score):")
        for i, group in enumerate(groups, 1):
            print(f"\nGroup {i} (similarity: {group.similarity_score:.4f}):")
            for value in sorted(group.values):
                print(f"  • {value}")
            
            # Let user choose canonical form
            print("\nChoose canonical form:")
            print("1. Select from existing values")
            print("2. Enter custom value")
            print("3. Skip this group")
            
            choice = input("Your choice (1-3): ").strip()
            
            if choice == "1":
                print("\nSelect from values:")
                for j, value in enumerate(sorted(group.values), 1):
                    print(f"{j}. {value}")
                try:
                    idx = int(input("Enter number: ").strip()) - 1
                    if 0 <= idx < len(sorted(group.values)):
                        canonical = sorted(group.values)[idx]
                        group.set_canonical_form(canonical)
                        print(f"Set canonical form to: {canonical}")
                except ValueError:
                    print("Invalid input, skipping group")
                    continue
            
            elif choice == "2":
                custom = input("Enter custom canonical form: ").strip()
                if custom:
                    group.set_canonical_form(custom)
                    print(f"Set canonical form to: {custom}")
            
            else:
                print("Skipping group")
                continue
        
        return groups
    
    async def run_tests(self):
        """Run consolidation tests on available datasets."""
        datasets = self._get_available_datasets()
        if not datasets:
            print(f"No CSV files found in {self.datasets_dir}")
            return
        
        print("\nAvailable datasets:")
        for i, dataset in enumerate(datasets, 1):
            df = pd.read_csv(os.path.join(self.datasets_dir, dataset))
            print(f"{i}. {dataset} ({len(df):,} rows)")
        
        try:
            print("\nSelect datasets:")
            print("0. Test all datasets")
            print("1-N. Enter dataset numbers (comma-separated) to test multiple datasets")
            dataset_input = input("Your choice: ").strip()
            
            # Get selected datasets
            if dataset_input == "0":
                # Test all datasets
                test_datasets = datasets
            else:
                # Test selected datasets
                dataset_indices = [int(idx.strip()) - 1 for idx in dataset_input.split(",")]
                test_datasets = [datasets[i] for i in dataset_indices]
            
            # Get sample sizes
            sample_sizes = {}
            print("\nEnter sample size for each dataset (press Enter for full dataset):")
            for dataset in test_datasets:
                df = pd.read_csv(os.path.join(self.datasets_dir, dataset))
                size_input = input(f"{dataset} ({len(df):,} rows) - Sample size: ").strip()
                sample_sizes[dataset] = int(size_input) if size_input else None
            
            # Ask if user wants to set canonical forms
            set_canonical = input("\nDo you want to set canonical forms for groups? (y/n): ").lower().strip() == 'y'
            
            for dataset_name in test_datasets:
                print(f"\nProcessing dataset: {dataset_name}")
                df = pd.read_csv(os.path.join(self.datasets_dir, dataset_name))
                
                # Sample dataset if requested
                sample_size = sample_sizes[dataset_name]
                if sample_size:
                    df = self._sample_dataset(df, sample_size)
                    print(f"Sampled {sample_size:,} rows from dataset of {len(df):,} total rows")
                
                # Show columns and get selection
                print("\nAvailable columns:")
                for i, col in enumerate(df.columns, 1):
                    print(f"{i}. {col} ({df[col].nunique()} unique values)")
                
                col_input = input("\nEnter column numbers (comma-separated, or 0 for all): ")
                if col_input.strip() == "0":
                    columns = df.columns.tolist()
                else:
                    col_indices = [int(idx.strip()) - 1 for idx in col_input.split(",")]
                    columns = [df.columns[i] for i in col_indices]
                
                # Process each column
                results = {
                    "dataset": dataset_name,
                    "total_rows": len(df),
                    "sampled_rows": sample_size,
                    "timestamp": datetime.now().isoformat(),
                    "column_results": []
                }
                
                for column in columns:
                    print(f"\nProcessing column: {column}")
                    column_info = self._process_column(df, column)
                    
                    # Run consolidation
                    groups = await self.agent.find_similar_values(df, column)
                    groups = [g for g in groups if len(g.values) > 1]
                    
                    # Set canonical forms if requested
                    if set_canonical:
                        print(f"\nSetting canonical forms for {column}:")
                        groups = self._select_canonical_forms(groups)
                    
                    column_info["groups"] = groups
                    results["column_results"].append(column_info)
                    print("-" * 50)
                
                # Save results
                self._save_results(dataset_name, results)
                
                # If canonical forms were set, ask if user wants to save consolidated dataset
                if set_canonical:
                    save_consolidated = input("\nDo you want to save the consolidated dataset? (y/n): ").lower().strip() == 'y'
                    if save_consolidated:
                        print("\nApplying consolidation mappings...")
                        
                        # Apply mappings to dataset
                        for column_result in results["column_results"]:
                            column = column_result["column"]
                            mappings = {}
                            for group in column_result["groups"]:
                                if group["canonical_form"]:
                                    for value in group["values"]:
                                        mappings[value] = group["canonical_form"]
                            
                            if mappings:
                                df[column] = df[column].map(lambda x: mappings.get(x, x))
                        
                        # Save consolidated dataset
                        consolidated_path = os.path.join(
                            self.results_dir, 
                            f"{dataset_name.replace('.csv', '')}_consolidated_{timestamp}.csv"
                        )
                        df.to_csv(consolidated_path, index=False)
                        print(f"\nConsolidated dataset saved to: {consolidated_path}")
                
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            return

async def main():
    test_bench = TestBench()
    await test_bench.run_tests()

if __name__ == "__main__":
    asyncio.run(main()) 