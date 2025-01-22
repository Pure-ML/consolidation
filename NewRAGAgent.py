import os
import pandas as pd
from typing import List, Dict, Set
from dataclasses import dataclass
import json
import logging
import tempfile
from datetime import datetime
from tqdm import tqdm

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

@dataclass
class ValueGroup:
    values: Set[str]
    canonical_form: str = ""
    match_explanation: str = ""

    def set_canonical_form(self, value: str):
        self.canonical_form = value
        logger.info(f"Set canonical form to: {value}")

class NewRAGAgent:
    def __init__(self, 
                 model: str = "gpt-4-turbo-preview",
                 embedding_model: str = "text-embedding-3-small"):
        logger.info(f"Initializing NewRAGAgent with model={model}, embedding_model={embedding_model}")
        self.llm = ChatOpenAI(model_name=model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.value_groups = []

    async def find_similar_values(self, df: pd.DataFrame, column: str) -> List[ValueGroup]:
        """Find groups of similar values in the specified column."""
        logger.info(f"Starting analysis of column: {column}")
        
        # 1. Extract unique values
        unique_vals = sorted(df[column].dropna().unique())
        logger.info(f"Found {len(unique_vals)} unique values")
        
        # 2. Create embeddings for all values at once
        logger.info("Creating embeddings for all values...")
        texts = [str(val) for val in unique_vals]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Create vector store using pre-computed embeddings
        logger.info("Creating vector store...")
        
        # Create a unique persistent directory
        persist_dir = os.path.join(tempfile.gettempdir(), f"chroma_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize Chroma with persistent directory and pre-computed embeddings
        import chromadb
        from chromadb.config import Settings
        
        # Create client with memory persistence
        chroma_client = chromadb.Client(Settings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Create collection
        collection = chroma_client.create_collection(name="similar_values")
        
        # Add documents with pre-computed embeddings
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"value": val} for val in unique_vals],
            ids=[str(i) for i in range(len(texts))]
        )
        
        # Create a custom retriever that uses the collection directly
        def get_similar_values(x: str) -> List[str]:
            # Find the ID for the query value
            try:
                query_idx = texts.index(str(x))
                query_id = str(query_idx)
                
                # Get similar values using pre-computed embeddings
                results = collection.query(
                    query_embeddings=[embeddings[query_idx]],
                    n_results=15,
                    include=['metadatas']
                )
                
                # Extract and return the similar values
                return [item['value'] for item in results['metadatas'][0]]
            except ValueError:
                logger.error(f"Value not found in texts: {x}")
                return []
        
        # 3. Create similarity search prompt
        prompt = ChatPromptTemplate.from_template(
            """You are analyzing text values to identify which ones represent the exact same entity but are written differently.
            Your task is to identify values that are DEFINITELY the same entity written in different ways.

            STRICT RULES - You MUST follow these exactly:
            1. ONLY group values that represent the EXACT SAME entity with different formatting:
               - Different punctuation or special characters
               - Different letter cases
               - Different separators or delimiters
               - Common misspellings or typos
               - Standard abbreviations of the exact same term
               - Universally accepted synonyms (e.g., exact equivalents)
               - Different but standardized representations of the same exact thing

            2. NEVER group values that:
               - Are merely related or similar
               - Share a category or classification
               - Are distinct entities
               - Have overlapping text but different meanings
               - Belong to the same family/type but are different entities

            For the value: {value}
            And its similar matches: {similar_values}

            Return ONLY a JSON object in this exact format:
            {{
                "should_group": true,
                "group_values": ["exact_value1", "exact_value1_variant"],
                "explanation": "Clear explanation of why these are formatting variations or exact equivalents of the same entity"
            }}

            or if no EXACT matches:
            {{
                "should_group": false
            }}

            DO NOT include any other text or explanation outside the JSON object.
            """
        )

        # 4. Process values and build groups
        processed_values = set()
        value_to_group = {}
        self.value_groups = []
        
        rag_chain = (
            {"value": RunnablePassthrough(), 
             "similar_values": get_similar_values}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        # Process each unique value
        for value in tqdm(unique_vals, desc="Processing values", unit="value"):
            if value in processed_values:
                logger.debug(f"Skipping already processed value: {value}")
                continue

            logger.info(f"Processing value: {value}")
            
            try:
                # Get similar values from vector search
                result = await rag_chain.ainvoke(value)
                
                # Clean up and parse the response
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]
                if result.endswith("```"):
                    result = result[:-3]
                
                try:
                    suggestion = json.loads(result.strip())
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response for value: {value}")
                    logger.error(f"Raw response: {result}")
                    logger.error(f"Error details: {str(e)}")
                    continue
                
                if suggestion.get("should_group", False):
                    group_values = set(suggestion["group_values"])
                    explanation = suggestion.get("explanation", "")
                    
                    # Only create or update groups if we have more than one value
                    if len(group_values) > 1:
                        # Check if any of these values belong to an existing group
                        existing_group_idx = None
                        for val in group_values:
                            if val in value_to_group:
                                existing_group_idx = value_to_group[val]
                                break
                        
                        if existing_group_idx is not None:
                            # Add to existing group
                            logger.info(f"Adding values to existing group {existing_group_idx}")
                            self.value_groups[existing_group_idx].values.update(group_values)
                            # Update mapping for all values
                            for val in group_values:
                                value_to_group[val] = existing_group_idx
                        else:
                            # Create new group
                            logger.info(f"Creating new group for: {group_values}")
                            new_group = ValueGroup(
                                values=group_values,
                                match_explanation=explanation
                            )
                            new_group_idx = len(self.value_groups)
                            self.value_groups.append(new_group)
                            # Map all values to this group
                            for val in group_values:
                                value_to_group[val] = new_group_idx
                        
                        processed_values.update(group_values)
                
            except Exception as e:
                logger.error(f"Error processing value {value}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Analysis complete. Found {len(self.value_groups)} groups of similar values")
        return self.value_groups

    def modify_group(self, group_idx: int, action: str, **kwargs) -> bool:
        """Modify a value group."""
        logger.info(f"Modifying group {group_idx} with action: {action}")
        
        if not (0 <= group_idx < len(self.value_groups)):
            logger.warning(f"Invalid group index: {group_idx}")
            return False

        group = self.value_groups[group_idx]
        
        if action == "remove_value":
            value = kwargs.get("value")
            if value in group.values:
                logger.info(f"Removing value '{value}' from group {group_idx}")
                group.values.remove(value)
                if len(group.values) < 2:
                    logger.info(f"Group {group_idx} now has less than 2 values, removing group")
                    self.value_groups.pop(group_idx)
                return True
            else:
                logger.warning(f"Value '{value}' not found in group {group_idx}")
        else:
            logger.warning(f"Unknown action: {action}")
        
        return False

    def split_group(self, group_idx: int, value: str) -> bool:
        """Split a value into a new group."""
        logger.info(f"Attempting to split value '{value}' from group {group_idx}")
        
        if not (0 <= group_idx < len(self.value_groups)):
            logger.warning(f"Invalid group index: {group_idx}")
            return False

        group = self.value_groups[group_idx]
        if value not in group.values:
            logger.warning(f"Value '{value}' not found in group {group_idx}")
            return False

        # Remove value from original group
        group.values.remove(value)
        logger.info(f"Removed '{value}' from group {group_idx}")
        
        # If original group now has less than 2 values, remove it
        if len(group.values) < 2:
            logger.info(f"Group {group_idx} now has less than 2 values, removing group")
            self.value_groups.pop(group_idx)
        
        # Create new group with just this value
        new_group = ValueGroup(values={value})
        self.value_groups.append(new_group)
        logger.info(f"Created new group for value '{value}'")
        
        return True

    def get_value_mappings(self) -> Dict[str, str]:
        """Get the final value mappings based on canonical forms."""
        logger.info("Generating final value mappings")
        mappings = {}
        for i, group in enumerate(self.value_groups):
            if group.canonical_form:
                for value in group.values:
                    mappings[value] = group.canonical_form
                logger.debug(f"Group {i}: Mapping {len(group.values)} values to '{group.canonical_form}'")
        
        logger.info(f"Generated {len(mappings)} total mappings")
        return mappings
