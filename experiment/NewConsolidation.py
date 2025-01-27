import os
import pandas as pd
from typing import List, Dict, Set
from dataclasses import dataclass
import json
import logging
import tempfile
from datetime import datetime
from tqdm import tqdm
import tiktoken
import asyncio
from experiment.NewRAGAgent import NewRAGAgent
from dotenv import load_dotenv
import numpy as np
import networkx as nx
from jellyfish import jaro_winkler_similarity
import re

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostTracker:
    # Current OpenAI pricing per 1K tokens (as of March 2024)
    COSTS = {
        "gpt-4": {
            "input": 0.03,
            "output": 0.06
        },
        "text-embedding-3-small": {
            "input": 0.00002,
            "output": 0.0  # No output tokens for embeddings
        }
    }

    def __init__(self):
        self.embedding_tokens = 0
        self.chat_input_tokens = 0
        self.chat_output_tokens = 0
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4 and text-embedding-3

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def add_embedding_tokens(self, text_list: List[str]):
        for text in text_list:
            self.embedding_tokens += self.count_tokens(str(text))

    def add_chat_tokens(self, input_text: str, output_text: str):
        self.chat_input_tokens += self.count_tokens(input_text)
        self.chat_output_tokens += self.count_tokens(output_text)

    def get_costs(self):
        embedding_cost = (self.embedding_tokens / 1000) * self.COSTS["text-embedding-3-small"]["input"]
        chat_input_cost = (self.chat_input_tokens / 1000) * self.COSTS["gpt-4"]["input"]
        chat_output_cost = (self.chat_output_tokens / 1000) * self.COSTS["gpt-4"]["output"]
        
        return {
            "embedding_tokens": self.embedding_tokens,
            "embedding_cost": embedding_cost,
            "chat_input_tokens": self.chat_input_tokens,
            "chat_output_tokens": self.chat_output_tokens,
            "chat_cost": chat_input_cost + chat_output_cost,
            "total_cost": embedding_cost + chat_input_cost + chat_output_cost
        }

    def log_costs(self):
        costs = self.get_costs()
        logger.info("=== OpenAI API Usage and Costs ===")
        logger.info(f"Embedding Tokens: {costs['embedding_tokens']:,} (${costs['embedding_cost']:.4f})")
        logger.info(f"Chat Input Tokens: {costs['chat_input_tokens']:,}")
        logger.info(f"Chat Output Tokens: {costs['chat_output_tokens']:,}")
        logger.info(f"Chat Total Cost: ${costs['chat_cost']:.4f}")
        logger.info(f"Total Cost: ${costs['total_cost']:.4f}")
        logger.info("================================")
        return costs

@dataclass
class ValueGroup:
    values: Set[str]
    canonical_form: str = ""
    match_explanation: str = ""

    def set_canonical_form(self, value: str):
        self.canonical_form = value
        logger.info(f"Set canonical form to: {value}")

def standardize_value(value: str) -> str:
    """Standardize a value by removing special chars, extra spaces, and converting to lowercase."""
    # Convert to string if not already
    value = str(value)
    # Remove special characters and extra whitespace
    value = re.sub(r'[^\w\s]', ' ', value)
    # Convert to lowercase and normalize whitespace
    value = ' '.join(value.lower().split())
    return value

class ImprovedRAGAgent:
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 model: str = "gpt-4",
                 embedding_model: str = "text-embedding-3-small",
                 batch_size: int = 5):  # Number of concurrent LLM calls
        self.similarity_threshold = similarity_threshold
        self.llm = ChatOpenAI(model_name=model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.cost_tracker = CostTracker()
        self.batch_size = batch_size
        
    async def _compute_similarities_batch(self, values: List[str], standardized_values: List[str], 
                                        start_idx: int, end_idx: int) -> List[tuple]:
        """Compute similarities for a batch of values."""
        edges = []
        for i in range(start_idx, min(end_idx, len(values))):
            for j in range(i + 1, len(values)):
                similarity = jaro_winkler_similarity(standardized_values[i], standardized_values[j])
                if similarity >= self.similarity_threshold:
                    edges.append((values[i], values[j], similarity))
        return edges
        
    async def _standardize_values_parallel(self, values: List[str], batch_size: int = 1000) -> List[str]:
        """Standardize values in parallel batches."""
        async def process_batch(batch: List[str]) -> List[str]:
            return [standardize_value(val) for val in batch]
        
        tasks = []
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            tasks.append(asyncio.create_task(process_batch(batch)))
        
        results = await asyncio.gather(*tasks)
        return [val for batch in results for val in batch]

    async def _get_embeddings_batch(self, values: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for values in parallel batches."""
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(values), batch_size):
            batch = values[i:i + batch_size]
            # Track token usage for embeddings
            self.cost_tracker.add_embedding_tokens(batch)
            # Generate embeddings for the batch
            batch_embeddings = await self.embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to respect rate limits
            if i + batch_size < len(values):
                await asyncio.sleep(0.1)
        
        return all_embeddings

    async def _build_similarity_graph_parallel(self, values: List[str], standardized_values: List[str]) -> nx.Graph:
        """Build similarity graph using parallel processing."""
        G = nx.Graph()
        
        # Pre-group values by their standardized form
        standardized_groups = {}
        for i, value in enumerate(values):
            std_value = standardized_values[i]
            if std_value not in standardized_groups:
                standardized_groups[std_value] = []
            standardized_groups[std_value].append(value)
        
        # Add one representative value per standardized form as node
        representative_values = []
        representative_std_values = []
        std_to_group = {}
        
        for std_value, group in standardized_groups.items():
            representative = group[0]  # Use first value as representative
            representative_values.append(representative)
            representative_std_values.append(std_value)
            std_to_group[representative] = set(group)
            G.add_node(representative, standardized=std_value, group=set(group))
        
        # Split work into batches based on CPU cores for similarity computation
        num_cpus = os.cpu_count() or 1
        batch_size = max(1, len(representative_values) // num_cpus)
        tasks = []
        
        # Create tasks for parallel similarity computation
        for start_idx in range(0, len(representative_values), batch_size):
            end_idx = start_idx + batch_size
            task = asyncio.create_task(
                self._compute_similarities_batch(representative_values, representative_std_values, start_idx, end_idx)
            )
            tasks.append(task)
        
        # Process batches in parallel and collect edges
        all_edges = []
        async with asyncio.TaskGroup() as tg:
            results = await asyncio.gather(*tasks)
            for batch_edges in results:
                all_edges.extend(batch_edges)
        
        # Add edges to graph
        G.add_weighted_edges_from(all_edges)
        return G

    async def _verify_groups_parallel(self, components: List[Set[str]], column_context: str) -> List[ValueGroup]:
        """Verify multiple groups in parallel using LLM."""
        verified_groups = []
        
        # Calculate optimal batch size based on group sizes
        async def get_optimal_batch(groups: List[Set[str]], max_tokens: int = 2000) -> List[Set[str]]:
            batch = []
            current_tokens = 0
            
            for group in groups:
                # Estimate tokens for this group (rough estimate: 4 tokens per word)
                group_tokens = sum(len(str(val).split()) * 4 for val in group)
                
                # If adding this group would exceed token limit, return current batch
                if current_tokens + group_tokens > max_tokens:
                    break
                
                batch.append(group)
                current_tokens += group_tokens
            
            return batch
        
        # Process components in dynamic batches
        remaining_components = [comp for comp in components if len(comp) > 1]
        while remaining_components:
            # Get next optimal batch
            batch = await get_optimal_batch(remaining_components)
            if not batch:
                # If we can't even fit one group, process it separately
                batch = [remaining_components[0]]
            
            # Remove processed groups from remaining
            remaining_components = remaining_components[len(batch):]
            
            # Verify batch
            results = await self._verify_batch_with_llm(batch, column_context)
            
            # Create groups for verified components
            for component, is_verified in zip(batch, results):
                if is_verified:
                    group = ValueGroup(values=set(component))
                    verified_groups.append(group)
                    logger.info(f"Found verified group with {len(component)} values")
            
            # Add small delay between batches to avoid rate limits
            if remaining_components:
                await asyncio.sleep(0.1)
        
        return verified_groups

    async def find_similar_values(self, df: pd.DataFrame, column: str) -> List[ValueGroup]:
        """Find groups of similar values using improved algorithm with parallel processing."""
        logger.info(f"Processing column: {column}")
        
        # Get column context
        sample_values = df[column].dropna().sample(min(5, len(df[column].dropna()))).tolist()
        column_context = f"""Column name: {column}
Sample values: {', '.join(map(str, sample_values))}
Total unique values: {df[column].nunique()}
Data type: {df[column].dtype}"""
        logger.info(f"Column context: {column_context}")
        
        # 1. Extract unique values
        unique_vals = sorted(df[column].dropna().unique())
        logger.info(f"Found {len(unique_vals)} unique values")
        
        # 2. Standardize values in parallel
        standardized_vals = await self._standardize_values_parallel(unique_vals)
        logger.info("Standardized all values")
        
        # 3. Get embeddings in parallel batches
        str_vals = [str(val) for val in unique_vals]
        embeddings = await self._get_embeddings_batch(str_vals)
        logger.info("Generated embeddings")
        
        # 4 & 5. Build similarity graph using parallel computation
        G = await self._build_similarity_graph_parallel(unique_vals, standardized_vals)
        logger.info(f"Built similarity graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # 6. Find connected components and expand groups to include all values
        raw_components = list(nx.connected_components(G))
        components = []
        for component in raw_components:
            expanded_component = set()
            for node in component:
                expanded_component.update(G.nodes[node]['group'])
            components.append(expanded_component)
        logger.info(f"Found {len(components)} connected components")
        
        # 7. Verify components with parallel LLM calls
        verified_groups = await self._verify_groups_parallel(components, column_context)
        
        logger.info(f"Final verified groups: {len(verified_groups)}")
        return verified_groups
    
    async def _verify_group_with_llm(self, values: Set[str]) -> bool:
        """Verify if a group of values truly represent the same entity using LLM."""
        prompt = f"""Do all these values represent EXACTLY the same entity, just written differently?
Values: {sorted(values)}

Rules:
1. Return TRUE only if ALL values are definitely the same entity with different formatting/spelling
2. Return FALSE if ANY values are:
   - Different entities
   - Just similar but not the same
   - Related but distinct

Return ONLY 'TRUE' or 'FALSE'."""
        
        response = await self.llm.ainvoke(prompt)
        self.cost_tracker.add_chat_tokens(prompt, response.content)
        return response.content.strip().upper() == 'TRUE'

    async def _verify_batch_with_llm(self, groups: List[Set[str]], column_context: str) -> List[bool]:
        """Verify multiple groups in a single LLM call."""
        # Calculate base prompt overhead
        base_prompt = f"""You are helping to consolidate similar values in a dataset to improve machine learning model performance.

Column Context:
{column_context}

Analyze each group of values and determine if they represent EXACTLY the same entity, just written differently.
For each group, determine if all values in that group are the same entity with different formatting/spelling.

Rules for each group:
1. Return TRUE only if ALL values are definitely the same entity with different formatting/spelling
2. Return FALSE if ANY values are:
   - Different entities
   - Just similar but not the same
   - Related but distinct

Return your response in the following JSON format:
{{
    "results": [true/false, ...],
    "explanation": "Brief explanation of any interesting decisions"
}}"""
        
        base_tokens = self.cost_tracker.count_tokens(base_prompt)
        # Leave room for response and some buffer
        max_batch_tokens = 3000  # Conservative limit to ensure we stay well under context limit
        
        current_batch = []
        current_batch_tokens = base_tokens
        all_results = []
        
        # Add groups to batches while keeping track of tokens
        for i, group in enumerate(groups, 1):
            values_str = ", ".join(sorted(group))
            group_text = f"Group {i}: {values_str}"
            group_tokens = self.cost_tracker.count_tokens(group_text)
            
            # If adding this group would exceed token limit, process current batch
            if current_batch_tokens + group_tokens > max_batch_tokens:
                if current_batch:
                    batch_results = await self._process_verification_batch(current_batch, column_context)
                    all_results.extend(batch_results)
                    current_batch = []
                    current_batch_tokens = base_tokens
            
            current_batch.append(group)
            current_batch_tokens += group_tokens
        
        # Process any remaining groups
        if current_batch:
            batch_results = await self._process_verification_batch(current_batch, column_context)
            all_results.extend(batch_results)
        
        return all_results

    async def _process_verification_batch(self, groups: List[Set[str]], column_context: str) -> List[bool]:
        """Process a single batch of groups for verification."""
        formatted_groups = []
        for i, group in enumerate(groups, 1):
            values_str = ", ".join(sorted(group))
            formatted_groups.append(f"Group {i}: {values_str}")
        
        prompt = f"""You are helping to consolidate similar values in a dataset to improve machine learning model performance.

Column Context:
{column_context}

Analyze each group of values ({len(groups)} groups total) and determine if they represent EXACTLY the same entity, just written differently.
For each group, determine if all values in that group are the same entity with different formatting/spelling.

{chr(10).join(formatted_groups)}

Rules for each group:
1. Return TRUE only if ALL values are definitely the same entity with different formatting/spelling
2. Return FALSE if ANY values are:
   - Different entities
   - Just similar but not the same
   - Related but distinct

Return your response in the following JSON format:
{{
    "results": [true/false, ...],
    "explanation": "Brief explanation of any interesting decisions"
}}

Example response:
{{
    "results": [true, false, true],
    "explanation": "Group 2 contains different products despite similar names"
}}"""

        # Estimate tokens before making the call
        estimated_tokens = self.cost_tracker.count_tokens(prompt)
        logger.debug(f"Estimated tokens for batch of {len(groups)} groups: {estimated_tokens}")
        
        response = await self.llm.ainvoke(prompt)
        self.cost_tracker.add_chat_tokens(prompt, response.content)
        
        try:
            # Parse the response as a JSON object
            parsed = json.loads(response.content.strip())
            if isinstance(parsed, dict) and "results" in parsed and isinstance(parsed["results"], list):
                if len(parsed["results"]) == len(groups):
                    if "explanation" in parsed and parsed["explanation"]:
                        logger.info(f"Batch verification note: {parsed['explanation']}")
                    return parsed["results"]
            
            logger.warning(f"Invalid LLM response structure: {response.content}")
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response.content}")
        
        # Fallback: verify each group individually
        logger.info("Falling back to individual group verification")
        results = []
        for group in groups:
            is_verified = await self._verify_group_with_llm(group)
            results.append(is_verified)
        return results
    
    def get_value_mappings(self, groups: List[ValueGroup]) -> Dict[str, str]:
        """Get the final value mappings based on canonical forms."""
        mappings = {}
        for group in groups:
            if group.canonical_form:
                for value in group.values:
                    mappings[value] = group.canonical_form
        return mappings

# Load environment variables
load_dotenv()

async def main():
    # Initialize the agent
    agent = ImprovedRAGAgent(
        model="gpt-4",
        embedding_model="text-embedding-3-large"
    )
    
    # Get input file path
    file_path = input("\nEnter CSV file path: ")
    df = pd.read_csv(file_path)
    
    # Print available columns
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col} ({df[col].nunique()} unique values)")
    
    # Get column choice
    col_idx = int(input("\nEnter column number: ")) - 1
    column = df.columns[col_idx]
    
    # Find similar values
    print(f"\nAnalyzing column: {column}")
    groups = await agent.find_similar_values(df, column)
    
    # Filter and display groups with more than 1 value
    groups = [g for g in groups if len(g.values) > 1]
    
    print(f"\nFound {len(groups)} groups of similar values:")
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i}:")
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

    # Show cost summary before asking to apply changes
    print("\nAPI Usage Summary:")
    costs = agent.cost_tracker.log_costs()
    print(f"Total Cost: ${costs['total_cost']:.4f}")

    # Ask user if they want to apply changes
    apply_changes = input("\nDo you want to apply these changes to the dataset? (y/n): ").lower().strip() == 'y'
    
    if apply_changes:
        # Get and apply mappings
        mappings = agent.get_value_mappings(groups)
        if mappings:
            print("\nProposed changes:")
            for original, new in sorted(mappings.items()):
                print(f"  {original} -> {new}")
            
            confirm = input("\nConfirm applying these changes? (y/n): ").lower().strip() == 'y'
            
            if confirm:
                df[column] = df[column].map(lambda x: mappings.get(x, x))
                print(f"\nApplied {len(mappings)} value mappings")
                
                # Ask for output path
                default_output = "consolidated_output.csv"
                output_path = input(f"\nEnter output path (default: {default_output}): ").strip()
                if not output_path:
                    output_path = default_output
                
                df.to_csv(output_path, index=False)
                print(f"Saved consolidated results to {output_path}")
            else:
                print("Changes not applied")
    else:
        print("No changes made to the dataset")

if __name__ == "__main__":
    asyncio.run(main())