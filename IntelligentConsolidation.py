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
from dotenv import load_dotenv
import numpy as np
import networkx as nx
from jellyfish import jaro_winkler_similarity
import re
import matplotlib.pyplot as plt
import random
import itertools

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
            "output": 0.0
        }
    }

    def __init__(self):
        self.embedding_tokens = 0
        self.chat_input_tokens = 0
        self.chat_output_tokens = 0
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
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
    mean_variance: float = 0.0  # Added to store group variance

    def set_canonical_form(self, value: str):
        self.canonical_form = value
        logger.info(f"Set canonical form to: {value}")

    def set_mean_variance(self, variance: float):
        self.mean_variance = variance
        logger.debug(f"Group variance: {variance:.4f}")

def standardize_value(value: str) -> str:
    """Standardize a value by removing all formatting, special chars, and spaces."""
    # Convert to string if not already
    value = str(value)
    # Remove all special characters including spaces
    value = re.sub(r'[^\w]', '', value)
    # Convert to lowercase
    value = value.lower()
    return value

class ImprovedRAGAgent:
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 model: str = "gpt-4",
                 embedding_model: str = "text-embedding-3-small",
                 batch_size: int = 5):  # Number of concurrent LLM calls
        self.similarity_threshold = similarity_threshold
        
        # Initialize OpenAI
        load_dotenv()
        self.llm = ChatOpenAI(model_name=model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        self.cost_tracker = CostTracker()
        self.batch_size = batch_size
        self.current_graph = None
        self.value_embeddings = {}  # Map of value to its embedding
        
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
        values_to_embed = []
        
        # First check which values we already have embeddings for
        for value in values:
            if value in self.value_embeddings:
                all_embeddings.append(self.value_embeddings[value])
            else:
                values_to_embed.append(value)
                
        if not values_to_embed:
            return all_embeddings
        
        # Process remaining values in batches
        for i in range(0, len(values_to_embed), batch_size):
            batch = values_to_embed[i:i + batch_size]
            # Track token usage for embeddings
            self.cost_tracker.add_embedding_tokens(batch)
            
            try:
                batch_embeddings = await self.embeddings.aembed_documents(batch)
                # Store embeddings for future use
                for val, emb in zip(batch, batch_embeddings):
                    self.value_embeddings[val] = emb
                all_embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(values_to_embed):
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                # Return zero embeddings as fallback
                return [[0.0] * 1536] * len(values)  # OpenAI embeddings are 1536-dimensional
        
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

    async def _calculate_group_variance(self, values: Set[str]) -> float:
        """Calculate the variance of embeddings within a group using cached embeddings."""
        if len(values) < 2:
            return 0.0
            
        # Get embeddings for all values (they should already be cached)
        embeddings = [self.value_embeddings[val] for val in values]
        embeddings_array = np.array(embeddings)
        
        # Calculate centroid
        centroid = np.mean(embeddings_array, axis=0)
        
        # Calculate distances from centroid
        distances = np.linalg.norm(embeddings_array - centroid, axis=1)
        
        # Log minimal statistics
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        logger.debug(f"Group variance stats: mean={mean_dist:.4f}, std={std_dist:.4f}")
        
        # Return variance of distances from centroid
        return float(np.var(distances))

    async def _analyze_and_split_group(self, values: Set[str], column_context: str) -> List[Set[str]]:
        """Analyze a rejected group and try to split it into valid subgroups."""
        prompt = f"""Based on this column context:
{column_context}

This group of values was determined NOT to be the same entity. Analyze if it can be split into smaller groups where each group contains values that ARE the same exact entity.

Values to analyze: {sorted(values)}

Instructions:
1. Look for subsets of values that represent exactly the same entity according to the column's purpose
2. Values can only be grouped if they are DEFINITELY the same exact entity written differently
3. Each value can only appear in one subgroup
4. Not every value needs to be in a subgroup
5. Only create subgroups if you are 100% certain the values are the same

Return your response in this EXACT format:
{{"subgroups": [
    ["value1", "value2"],  // First group of same entity
    ["value3", "value4"]   // Second group of same entity
]}}

If no valid subgroups exist, return exactly: {{"subgroups": []}}

Remember: Only group values if they refer to the SAME EXACT entity according to the column's purpose."""

        response = await self._llm_generate(prompt)
        
        try:
            # Parse the response
            result = json.loads(response.strip())
            if not isinstance(result, dict) or "subgroups" not in result:
                logger.warning(f"Invalid response format: {response}")
                return []
                
            # Convert lists to sets
            subgroups = [set(subgroup) for subgroup in result["subgroups"]]
            
            # Validate subgroups
            all_values = set()
            for subgroup in subgroups:
                if not subgroup.issubset(values) or subgroup & all_values:
                    logger.warning("Invalid subgroups detected: overlapping or invalid values")
                    return []
                all_values.update(subgroup)
            
            valid_subgroups = [subgroup for subgroup in subgroups if len(subgroup) > 1]
            if valid_subgroups:
                logger.info(f"Found {len(valid_subgroups)} valid subgroups:")
                for i, subgroup in enumerate(valid_subgroups, 1):
                    logger.info(f"  Subgroup {i}: {sorted(subgroup)}")
            else:
                logger.info("No valid subgroups found")
            
            return valid_subgroups
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse subgroups: {e}")
            return []

    async def _verify_groups_parallel(self, components: List[Set[str]], column_context: str) -> List[ValueGroup]:
        """Verify groups using batched LLM verification."""
        verified_groups = []
        
        # Components passed in should already be multi-node only
        if not components:
            return verified_groups
            
        # Verify all components with LLM
        logger.info(f"Sending {len(components)} groups for LLM verification")
        results = await self._verify_batch_with_llm(components, column_context)
        
        # Process each component
        for component, is_verified in zip(components, results):
            if is_verified:
                group = ValueGroup(values=set(component))
                variance = await self._calculate_group_variance(component)
                group.set_mean_variance(variance)
                verified_groups.append(group)
                logger.info(f"LLM verified group with {len(component)} values (variance: {variance:.4f})")
            else:
                # Try to split rejected group into valid subgroups
                logger.info(f"Analyzing rejected group of size {len(component)} for potential subgroups")
                subgroups = await self._analyze_and_split_group(component, column_context)
                
                if subgroups:
                    logger.info(f"Found {len(subgroups)} valid subgroups from rejected group")
                    for subgroup in subgroups:
                        group = ValueGroup(values=subgroup)
                        variance = await self._calculate_group_variance(subgroup)
                        group.set_mean_variance(variance)
                        verified_groups.append(group)
                        logger.info(f"Added subgroup with {len(subgroup)} values (variance: {variance:.4f})")
        
        # Sort groups by variance for better visualization
        verified_groups.sort(key=lambda x: x.mean_variance)
        return verified_groups

    async def _get_column_description(self, df: pd.DataFrame, column: str) -> str:
        """Get a concise description of what the column represents."""
        # Get a minimal sample of values
        sample_size = min(5, len(df[column].dropna()))
        sample_values = df[column].dropna().sample(sample_size).tolist()
        
        prompt = f"""Column: {column}
Values: {', '.join(map(str, sample_values))}
Type: {df[column].dtype}

What does this column represent? One sentence only."""

        response = await self._llm_generate(prompt)
        return response.strip()

    async def find_similar_values(self, df: pd.DataFrame, column: str) -> List[ValueGroup]:
        """Find groups of similar values using parallel processing."""
        logger.info(f"Processing column: {column}")
        
        # Get minimal column context
        column_context = f"""Column: {column}
Type: {df[column].dtype}
Values: {', '.join(map(str, df[column].dropna().sample(min(3, len(df[column].dropna()))).tolist()))}"""
        logger.info(f"Column context: {column_context}")
        
        # 1. Get column context with LLM description
        column_description = await self._get_column_description(df, column)
        sample_values = df[column].dropna().sample(min(5, len(df[column].dropna()))).tolist()
        column_context = f"""Column name: {column}
Description: {column_description}
Sample values: {', '.join(map(str, sample_values))}
Total unique values: {df[column].nunique()}
Data type: {df[column].dtype}"""
        logger.info(f"Column context: {column_context}")
        
        # 2. Extract unique values
        unique_vals = sorted(df[column].dropna().unique())
        logger.info(f"Found {len(unique_vals)} unique values")
        
        # 3 & 4. Parallelize embeddings and standardization
        # Run both operations concurrently
        embedding_task = asyncio.create_task(self._get_embeddings_batch([str(val) for val in unique_vals]))
        standardize_task = asyncio.create_task(self._standardize_values_parallel(unique_vals))
        
        # Wait for both tasks to complete
        standardized_vals = await standardize_task
        _ = await embedding_task  # We don't need embeddings yet, but we start them early
        logger.info("Completed standardization and embeddings")
        
        # 5. Form initial groups from exact standardized matches
        std_to_original = {}
        for i, val in enumerate(unique_vals):
            std_val = standardized_vals[i]
            if std_val not in std_to_original:
                std_to_original[std_val] = set()
            std_to_original[std_val].add(val)
        
        # Initial groups are those with multiple values for same standardized form
        initial_groups = [vals for vals in std_to_original.values() if len(vals) > 1]
        logger.info(f"Found {len(initial_groups)} initial groups from exact standardized matches")
        
        # 6 & 7. Build similarity graph using unique standardized values
        unique_std_vals = list(std_to_original.keys())
        G = await self._build_similarity_graph_parallel(unique_std_vals, unique_std_vals)
        logger.info(f"Built similarity graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Store for visualization
        self.current_graph = G
        self.visualize_graph()
        
        # 8. Find connected components and expand to include all original values
        raw_components = [comp for comp in nx.connected_components(G) if len(comp) > 1]
        components = []
        for component in raw_components:
            expanded_component = set()
            for std_val in component:
                expanded_component.update(std_to_original[std_val])
            components.append(expanded_component)
        logger.info(f"Found {len(components)} multi-node components")
        
        # 9, 10, 11. Verify components with LLM and handle subgroups
        verified_groups = await self._verify_groups_parallel(components, column_context)
        logger.info(f"Final verified groups: {len(verified_groups)}")
        
        # Add initial exact-match groups that weren't part of graph components
        component_values = {val for comp in components for val in comp}
        for group in initial_groups:
            if not any(val in component_values for val in group):
                group_obj = ValueGroup(values=group)
                variance = await self._calculate_group_variance(group)
                group_obj.set_mean_variance(variance)
                verified_groups.append(group_obj)
        
        # Sort all groups by variance for better presentation
        verified_groups.sort(key=lambda x: x.mean_variance)
        return verified_groups
    
    async def _verify_batch_with_llm(self, components: List[Set[str]], column_context: str) -> List[bool]:
        """Verify multiple groups in parallel batches."""
        # Constants for batching and rate limiting
        MAX_BATCH_SIZE = 10      # Maximum groups per batch
        
        all_results = []
        
        # Process components in batches
        for i in range(0, len(components), MAX_BATCH_SIZE):
            batch = components[i:i + MAX_BATCH_SIZE]
            batch_results = await self._process_verification_batch(batch, column_context)
            all_results.extend(batch_results)
            
            # Add delay between batches
            if i + MAX_BATCH_SIZE < len(components):
                await asyncio.sleep(0.1)
        
        return all_results

    async def _verify_group_with_llm(self, values: Set[str], column_context: str) -> bool:
        """Verify if a group of values truly represent the same entity using LLM."""
        prompt = f"""Based on this column context:
{column_context}

Analyze if these values represent the SAME EXACT entity, just written differently:
{sorted(values)}

Rules:
1. Return TRUE only if ALL values refer to the SAME EXACT entity written in different ways
2. Return FALSE if ANY values are:
   - Different entities
   - Similar but not identical entities
   - Related but distinct entities

Return ONLY 'TRUE' or 'FALSE'."""
        
        response = await self._llm_generate(prompt)
        return response.strip().upper() == 'TRUE'

    async def _llm_generate(self, prompt: str) -> str:
        """Helper function to generate text with GPT-4."""
        try:
            response = await self.llm.ainvoke(prompt)
            text = response.content
            self.cost_tracker.add_chat_tokens(prompt, text)
            return text
        except Exception as e:
            logger.error(f"Error generating with GPT-4: {e}")
            return ""

    def get_value_mappings(self, groups: List[ValueGroup]) -> Dict[str, str]:
        """Get the final value mappings based on canonical forms."""
        mappings = {}
        for group in groups:
            if group.canonical_form:
                for value in group.values:
                    mappings[value] = group.canonical_form
        return mappings

    def visualize_graph(self, output_path: str = "similarity_graph.png"):
        """Visualize the similarity graph with components in different colors."""
        if self.current_graph is None:
            logger.warning("No graph available to visualize")
            return

        plt.figure(figsize=(15, 10))
        
        # Get components for coloring
        components = list(nx.connected_components(self.current_graph))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(components)))
        
        # Create position layout
        pos = nx.spring_layout(self.current_graph, k=1, iterations=50)
        
        # Draw nodes for each component with different colors
        for idx, component in enumerate(components):
            nx.draw_networkx_nodes(self.current_graph, pos,
                                 nodelist=list(component),
                                 node_color=[colors[idx]],
                                 node_size=1000,
                                 alpha=0.6)
        
        # Draw edges with weights as labels
        edge_weights = nx.get_edge_attributes(self.current_graph, 'weight')
        nx.draw_networkx_edge_labels(self.current_graph, pos, 
                                   edge_labels={e: f'{w:.2f}' for e, w in edge_weights.items()})
        nx.draw_networkx_edges(self.current_graph, pos)
        
        # Draw node labels
        nx.draw_networkx_labels(self.current_graph, pos)
        
        plt.title(f"Similarity Graph (threshold={self.similarity_threshold})\n{len(components)} components")
        plt.axis('off')
        
        # Save the plot
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Graph visualization saved to {output_path}")
        
        # Print component details
        print(f"\nFound {len(components)} components:")
        for idx, component in enumerate(components, 1):
            print(f"\nComponent {idx} ({len(component)} nodes):")
            for node in sorted(component):
                group = self.current_graph.nodes[node].get('group', {node})
                print(f"  • {node} (group: {sorted(group)})")

    async def _process_verification_batch(self, groups: List[Set[str]], column_context: str) -> List[bool]:
        """Process a batch of groups for verification."""
        # Format each group for analysis
        formatted_groups = []
        for i, group in enumerate(groups, 1):
            values_str = ", ".join(sorted(group))
            formatted_groups.append(f"Group {i}: {values_str}")
        
        prompt = f"""Based on this column context:
{column_context}

Analyze each group of values and determine if they represent the SAME EXACT entity according to the column's purpose.

{chr(10).join(formatted_groups)}

Rules:
1. Return TRUE only if ALL values in a group refer to the SAME EXACT entity written in different ways
2. Return FALSE if ANY values in a group are:
   - Different entities
   - Similar but not identical entities
   - Related but distinct entities

Return a JSON array of boolean values, one for each group, in order. Example: [true, false, true]"""
        
        # Estimate tokens before making the call
        estimated_tokens = self.cost_tracker.count_tokens(prompt)
        logger.debug(f"Estimated tokens for batch of {len(groups)} groups: {estimated_tokens}")
        
        response = await self._llm_generate(prompt)
        
        try:
            # Parse the response as a JSON array
            results = json.loads(response.strip())
            if not isinstance(results, list) or len(results) != len(groups):
                logger.warning(f"Invalid LLM response format: {response}")
                # Fallback: verify each group individually
                results = []
                for group in groups:
                    is_verified = await self._verify_group_with_llm(group, column_context)
                    results.append(is_verified)
            return results
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {response}")
            # Fallback: verify each group individually
            results = []
            for group in groups:
                is_verified = await self._verify_group_with_llm(group, column_context)
                results.append(is_verified)
            return results

# Load environment variables
load_dotenv()

async def main():
    # Initialize the agent
    agent = ImprovedRAGAgent(
        model="gpt-4",
        embedding_model="text-embedding-3-small"
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
    
    # Ask if user wants to save graph with custom filename
    custom_path = input("\nEnter path to save graph visualization (press Enter for default 'similarity_graph.png'): ").strip()
    if custom_path:
        agent.visualize_graph(custom_path)
    
    # Filter and display groups with more than 1 value
    groups = [g for g in groups if len(g.values) > 1]
    
    print(f"\nFound {len(groups)} groups of similar values (sorted by variance):")
    for i, group in enumerate(groups, 1):
        print(f"\nGroup {i} (variance: {group.mean_variance:.4f}):")
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