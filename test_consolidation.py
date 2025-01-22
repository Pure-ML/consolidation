import pandas as pd
from ConsolidationAgent import ConsolidationAgent, SimilarityConfig, TextPattern
import asyncio
from typing import Dict, List, Set
from readable_mapping_script import reorganize_mapping
import json

async def process_column(df: pd.DataFrame, column: str, agent: ConsolidationAgent) -> Dict[str, str]:
    """Process a single column and return the value mappings."""
    print(f"\nAnalyzing column: {column}")
    
    # Display current unique values
    unique_vals = sorted(df[column].dropna().unique())
    print(f"\nCurrent unique values ({len(unique_vals)}):")
    for val in unique_vals:
        print(f"  - {val}")
    
    # Find similar values
    print("\nFinding similar values...")
    groups = await agent.find_similar_values(df, column)
    
    value_mappings = {}
    if not groups:
        print("\nNo similar value groups found.")
        return value_mappings
    
    print(f"\nFound {len(groups)} groups of similar values:")
    for i, group in enumerate(groups, 1):
        while True:  # Loop for group modifications
            print(f"\nGroup {i}:")
            sorted_values = sorted(group.values)
            print("Values:")
            for j, value in enumerate(sorted_values, 1):
                print(f"   {j}. {value}")
            if group.match_explanation:
                print(f"Match explanation: {group.match_explanation}")
            
            print("\nOptions:")
            print("1. Set canonical form")
            print("2. Remove value from group")
            print("3. Split value into new group")
            print("4. Skip this group")
            print("5. Accept group as is")
            
            choice = input("\nYour choice (1-5): ").strip()
            
            if choice == "1":
                print("\nSet canonical form:")
                print("1. Select from existing values:")
                for j, value in enumerate(sorted_values, 1):
                    print(f"   {j}. {value}")
                print("2. Enter custom value")
                
                form_choice = input("\nYour choice (1 or 2): ").strip()
                try:
                    if form_choice == "1":
                        idx = int(input("Enter number: ").strip()) - 1
                        if 0 <= idx < len(sorted_values):
                            group.set_canonical_form(sorted_values[idx])
                            break
                    elif form_choice == "2":
                        custom = input("Enter custom canonical form: ").strip()
                        if custom:
                            group.canonical_form = custom
                            break
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Invalid input")
            
            elif choice == "2":
                print("\nSelect value to remove:")
                for j, value in enumerate(sorted_values, 1):
                    print(f"   {j}. {value}")
                try:
                    idx = int(input("Enter number (0 to cancel): ").strip()) - 1
                    if idx == -1:  # User cancelled
                        continue
                    if 0 <= idx < len(sorted_values):
                        if agent.modify_group(i-1, "remove_value", value=sorted_values[idx]):
                            if not group.values:  # Group was removed
                                break
                            group = agent.value_groups[i-1]  # Refresh group after modification
                except ValueError:
                    print("Invalid input")
            
            elif choice == "3":
                print("\nSelect value to split into new group:")
                for j, value in enumerate(sorted_values, 1):
                    print(f"   {j}. {value}")
                try:
                    idx = int(input("Enter number (0 to cancel): ").strip()) - 1
                    if idx == -1:  # User cancelled
                        continue
                    if 0 <= idx < len(sorted_values):
                        if agent.split_group(i-1, sorted_values[idx]):
                            if not group.values:  # Group was removed
                                break
                            group = agent.value_groups[i-1]  # Refresh group after modification
                except ValueError:
                    print("Invalid input")
            
            elif choice == "4":
                break
            
            elif choice == "5":
                # If no canonical form is set, use the first value
                if not group.canonical_form:
                    group.set_canonical_form(sorted_values[0])
                break
            
            else:
                print("Invalid choice")
        
        # Add mappings for this group if it has a canonical form
        if group and group.canonical_form:
            for value in group.values:
                value_mappings[value] = group.canonical_form
    
    # Update metrics based on final mappings
    if value_mappings:
        agent.metrics.final_unique_count = len(unique_vals) - len(value_mappings) + len(set(value_mappings.values()))
        print(f"\nFinal consolidation metrics:")
        print(f"{agent.get_metrics()}")
    
    return value_mappings

async def main():
    # Load the dataset

    # # modified cars dataset 
    # df = pd.read_csv("modified_inconsistent_dataset.csv")


    # job-catergorisation datatset FIRST 5000 ROWS
    df = pd.read_csv("job-categorisation.csv").head(300)
    
    
    # Initialize the agent
    agent = ConsolidationAgent()
    
    # Display available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    
    # Process columns until user is done
    processed_df = df.copy()
    while True:
        try:
            col_idx = int(input("\nEnter column number to process (0 to finish): ").strip()) - 1
            if col_idx == -1:
                break
            if 0 <= col_idx < len(df.columns):
                column = df.columns[col_idx]
                mappings = await process_column(processed_df, column, agent)

                # print the mappings
                print(f"Mappings for column {column}:")
                organized_mapping = reorganize_mapping(mappings)
                print(organized_mapping)
                # save reorganized mapping dictionary to a json file
                with open("reorganized_mapping.json", "w") as f:
                    json.dump(organized_mapping, f)
                
                # Apply mappings to the dataframe
                if mappings:
                    processed_df[column] = processed_df[column].map(lambda x: mappings.get(x, x))
                    print(f"\nApplied {len(mappings)} value mappings to column '{column}'")
            else:
                print("Invalid column number")
        except ValueError:
            print("Invalid input")
    
    # Save the processed dataframe if any changes were made
    if not processed_df.equals(df):
        output_path = "consolidated_dataset.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"\nSaved consolidated dataset to {output_path}")
    else:
        print("\nNo changes were made to the dataset")

if __name__ == "__main__":
    asyncio.run(main()) 