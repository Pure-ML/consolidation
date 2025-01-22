from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from pydantic import BaseModel, Field
from thefuzz import fuzz
from collections import defaultdict
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json
import tiktoken
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class ConsolidationMetrics:
    """Tracks metrics for the consolidation process."""
    initial_unique_count: int = 0
    final_unique_count: int = 0
    total_groups_found: int = 0
    llm_groups_found: int = 0
    algorithmic_groups_found: int = 0
    
    @property
    def reduction_percentage(self) -> float:
        """Calculate the percentage reduction in unique values."""
        if self.initial_unique_count == 0:
            return 0.0
        reduction = self.initial_unique_count - self.final_unique_count
        return (reduction / self.initial_unique_count) * 100
    
    def __str__(self) -> str:
        """Format metrics for display."""
        return f"""Consolidation Metrics:
• Initial unique values: {self.initial_unique_count}
• Final unique values: {self.final_unique_count}
• Total reduction: {self.reduction_percentage:.1f}%
• Groups detected: {self.total_groups_found}
  - LLM groups: {self.llm_groups_found}
  - Algorithmic groups: {self.algorithmic_groups_found}"""

class TextUtils:
    """Utility class for text processing."""
    @staticmethod
    def clean_text(text: str) -> str:
        """Standardize text for comparison."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Remove special characters and standardize spacing
        cleaned = re.sub(r'[./\\-_]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.lower().strip()
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate number of tokens in text using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            # Fallback: rough estimate based on words and punctuation
            return len(text.split()) + len(re.findall(r'[^\w\s]', text))

class BatchProcessor:
    """Handles batching of values for efficient LLM processing."""
    # MAX_TOKENS_PER_BATCH = 4000  # Conservative limit for context window
    MAX_TOKENS_PER_BATCH = 1000  # Conservative limit for context window

    @staticmethod
    def create_batches(values: Set[str]) -> List[Set[str]]:
        """Split values into batches based on token count."""
        sorted_values = sorted(values)  # Sort for consistent batching
        batches = []
        current_batch = set()
        current_tokens = 0
        
        base_prompt_tokens = TextUtils.estimate_tokens(
            "Looking at these values, group the synonymous entries together and explain why they are grouped:"
        )
        
        for value in sorted_values:
            # Estimate tokens for this value (including formatting)
            value_tokens = TextUtils.estimate_tokens(f"  - {value}\n")
            
            # Check if adding this value would exceed token limit
            if current_tokens + value_tokens + base_prompt_tokens > BatchProcessor.MAX_TOKENS_PER_BATCH:
                if current_batch:  # Only add non-empty batches
                    batches.append(current_batch)
                current_batch = {value}
                current_tokens = value_tokens + base_prompt_tokens
            else:
                current_batch.add(value)
                current_tokens += value_tokens
        
        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches

class TextPattern(BaseModel):
    """Pattern for text variation detection."""
    pattern_type: str = Field(description="Type of pattern (e.g., 'abbreviation', 'format', 'prefix')")
    description: str = Field(description="Description of the pattern")
    threshold: float = Field(description="Similarity threshold for this pattern", ge=0.0, le=100.0)

class SimilarityConfig(BaseModel):
    """Configuration for similarity detection."""
    token_threshold: float = Field(
        default=85.0,
        description="Threshold for token-based similarity",
        ge=0.0,
        le=100.0
    )
    partial_ratio_threshold: float = Field(
        default=90.0,
        description="Threshold for partial string matching",
        ge=0.0,
        le=100.0
    )
    use_llm: bool = Field(
        default=True,
        description="Whether to use LLM for semantic grouping"
    )
    patterns: List[TextPattern] = Field(
        default_factory=lambda: [
            TextPattern(
                pattern_type="format",
                description="Different formatting of same text",
                threshold=90.0
            ),
            TextPattern(
                pattern_type="prefix_suffix",
                description="One text is a prefix/suffix of another",
                threshold=80.0
            ),
            TextPattern(
                pattern_type="abbreviation",
                description="Abbreviated forms of text",
                threshold=75.0
            )
        ]
    )

class ValueGroup:
    """A group of similar/equivalent values."""
    def __init__(self, values: Set[str]):
        self.values = values
        self._canonical_form: Optional[str] = None
        self.match_explanation: Optional[str] = None
        self.similarity_scores: Dict[Tuple[str, str], float] = {}  # Store similarity scores between pairs
    
    @property
    def canonical_form(self) -> Optional[str]:
        return self._canonical_form
    
    @canonical_form.setter
    def canonical_form(self, value: Optional[str]):
        """Set canonical form, allowing custom values."""
        self._canonical_form = value
    
    def set_canonical_form(self, value: str):
        """Set canonical form from existing values."""
        if value in self.values:
            self._canonical_form = value
        else:
            raise ValueError(f"Canonical form must be one of: {self.values}")
    
    def add_similarity_score(self, value1: str, value2: str, score: float):
        """Add similarity score between two values."""
        self.similarity_scores[(value1, value2)] = score
        self.similarity_scores[(value2, value1)] = score
    
    def get_similarity_score(self, value1: str, value2: str) -> Optional[float]:
        """Get similarity score between two values."""
        return self.similarity_scores.get((value1, value2))
    
    def update_values(self, values: Set[str]):
        """Update the group's values while preserving the canonical form."""
        old_canonical = self._canonical_form
        self.values = values
        if old_canonical and old_canonical not in values:
            self._canonical_form = None
    
    def remove_value(self, value: str) -> bool:
        """Remove a value from the group, returns True if successful."""
        if value in self.values:
            if len(self.values) <= 2:  # If only 2 values, dissolve the group
                return False
            self.values.remove(value)
            if self._canonical_form == value:
                self._canonical_form = None
            # Remove similarity scores involving this value
            self.similarity_scores = {
                k: v for k, v in self.similarity_scores.items()
                if value not in k
            }
            return True
        return False
    
    def __str__(self) -> str:
        values_str = " -> ".join(sorted(self.values))
        explanation = f" ({self.match_explanation})" if self.match_explanation else ""
        return f"{values_str}{explanation}"

class ConsolidationAgent:
    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        self.value_groups: List[ValueGroup] = []
        self.metrics = ConsolidationMetrics()
        self.similarity_cache: Dict[Tuple[str, str], Tuple[float, str, str]] = {}
        
        # Initialize OpenAI if using LLM
        if self.config.use_llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = AsyncOpenAI(api_key=api_key)
    
    async def _get_llm_groups(self, values: Set[str]) -> List[ValueGroup]:
        """Use LLM to identify groups of similar values with batch processing."""
        all_groups = []
        
        # Split values into batches
        batches = BatchProcessor.create_batches(values)

        total_batches = len(batches)
        # print the total number of batches
        print(f"Total batches: {total_batches}")
        
        for i, batch in enumerate(batches, 1):
            if total_batches > 1:
                print(f"\nProcessing batch {i}/{total_batches} ({len(batch)} values)...")
            
            prompt = f"""Analyze these values and identify groups of equivalent entries that represent the same entity but with different formatting, spelling, or naming conventions.

Values:
{sorted(batch)}

Return the groups in JSON format:
{{
    "groups": [
        {{
            "values": ["value1", "value2", "value3"],
            "explanation": "Brief reason why these are the same",
            "confidence": "HIGH/MEDIUM/LOW"
        }},
        ...
    ]
}}

Guidelines:
- Group entries that clearly represent the same entity
- Consider variations in spelling, formatting, and common abbreviations
- Include typos and minor formatting differences
- For product names, group different versions/formats if they're clearly the same base product
- Provide clear explanations for why values are grouped
- Indicate confidence level for each grouping"""
            
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a data cleaning expert specializing in identifying equivalent values and inconsistencies in datasets."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content)
                batch_groups = []
                for group_info in result["groups"]:
                    group = ValueGroup(set(group_info["values"]))
                    group.match_explanation = f"{group_info['explanation']} (Confidence: {group_info.get('confidence', 'MEDIUM')})"
                    batch_groups.append(group)
                
                # Merge with existing groups if they share values
                all_groups = self._merge_groups(all_groups, batch_groups)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in batch {i}, attempting to fix response format...")
                try:
                    # Try to clean and repair the JSON
                    content = response.choices[0].message.content
                    content = re.sub(r'\\[^"\\\/bfnrtu]', '', content)  # Remove invalid escapes
                    result = json.loads(content)
                    # Process the cleaned result...
                except Exception as e2:
                    print(f"Failed to recover from JSON error: {str(e2)}")
            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
        
        self.metrics.llm_groups_found = len(all_groups)
        return all_groups
    
    def _merge_groups(self, existing_groups: List[ValueGroup], new_groups: List[ValueGroup]) -> List[ValueGroup]:
        """Merge groups that share common values."""
        if not existing_groups:
            return new_groups
        
        # Create a mapping of values to their group index
        value_to_group = {}
        for i, group in enumerate(existing_groups):
            for value in group.values:
                value_to_group[value] = i
        
        # Process each new group
        for new_group in new_groups:
            # Find all existing groups that share values with this new group
            related_groups = set()
            for value in new_group.values:
                if value in value_to_group:
                    related_groups.add(value_to_group[value])
            
            if not related_groups:
                # No overlap, add as new group
                existing_groups.append(new_group)
            else:
                # Merge with existing groups
                target_group = min(related_groups)
                merged_values = existing_groups[target_group].values.union(new_group.values)
                merged_explanation = f"{existing_groups[target_group].match_explanation}; {new_group.match_explanation}"
                
                # Update the target group
                existing_groups[target_group].values = merged_values
                existing_groups[target_group].match_explanation = merged_explanation
                
                # Remove other related groups
                existing_groups = [g for i, g in enumerate(existing_groups) if i not in related_groups - {target_group}]
                
                # Update value_to_group mapping
                for value in merged_values:
                    value_to_group[value] = target_group
        
        return existing_groups
    
    def _clean_text(self, text: str) -> str:
        """Standardize text for comparison."""
        return TextUtils.clean_text(text)
    
    def _get_variations(self, text: str) -> Set[str]:
        """Generate potential variations of the text for comparison."""
        variations = {text}
        cleaned = self._clean_text(text)
        variations.add(cleaned)
        
        # Handle common text variations
        base_form = cleaned.lower()
        
        # Remove common prefixes
        prefixes = ['v', 'the']
        for prefix in prefixes:
            if base_form.startswith(prefix):
                variations.add(base_form[len(prefix):].strip())
        
        # Handle special characters and formats
        no_special = re.sub(r'[./\\-_]', '', base_form)
        variations.add(no_special)
        
        # Handle multi-word names
        words = base_form.split()
        if len(words) > 1:
            # Add first word only (common in many domains)
            variations.add(words[0])
            
            # Add without common suffixes
            suffixes = ['inc', 'incorporated', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited']
            for suffix in suffixes:
                if base_form.endswith(suffix):
                    variations.add(base_form[:-len(suffix)].strip())
            
            # Add abbreviation if meaningful
            if len(words) <= 5:  # Only for reasonably short phrases
                abbrev = ''.join(word[0] for word in words)
                if len(abbrev) > 1:  # Only add if meaningful
                    variations.add(abbrev)
        
        return variations
    
    def _calculate_similarity(self, str1: str, str2: str) -> Tuple[float, str, Optional[str]]:
        """Calculate similarity between two strings and identify the type of match."""
        cache_key = (str1, str2)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get variations for both strings
        variations1 = self._get_variations(str1)
        variations2 = self._get_variations(str2)
        
        # Check for exact matches between variations
        if variations1 & variations2:
            result = (100.0, "variation", "Text variations or abbreviations")
            self.similarity_cache[cache_key] = result
            return result
        
        # Get base forms for comparison
        base1 = self._clean_text(str1)
        base2 = self._clean_text(str2)
        
        # Calculate various similarity metrics
        token_sort_ratio = fuzz.token_sort_ratio(base1, base2)
        token_set_ratio = fuzz.token_set_ratio(base1, base2)
        partial_ratio = fuzz.partial_ratio(base1, base2)
        
        # Enhanced similarity detection
        max_score = max(token_sort_ratio, token_set_ratio, partial_ratio)
        
        # Check for high token similarity (for handling word rearrangements)
        if token_set_ratio >= self.config.token_threshold:
            result = (token_set_ratio, "token", "Similar word patterns")
            self.similarity_cache[cache_key] = result
            return result
        
        # Check for prefix/suffix relationships with improved scoring
        if (base1.startswith(base2) or base2.startswith(base1) or
            base1.endswith(base2) or base2.endswith(base1)):
            score = max(partial_ratio, token_sort_ratio)
            if score >= self.config.patterns[1].threshold:
                result = (score, "prefix_suffix", "One value is a prefix/suffix of another")
                self.similarity_cache[cache_key] = result
                return result
        
        # Check for partial matches (useful for abbreviated forms)
        if partial_ratio >= self.config.partial_ratio_threshold:
            result = (partial_ratio, "partial", "Partial string match")
            self.similarity_cache[cache_key] = result
            return result
        
        # Check for common typos and character swaps
        if max_score >= 70:  # Lower threshold for potential typos
            char_diff = sum(1 for a, b in zip(base1, base2) if a != b)
            if char_diff <= 2 and abs(len(base1) - len(base2)) <= 2:
                result = (max_score, "typo", "Possible typo or character swap")
                self.similarity_cache[cache_key] = result
                return result
        
        result = (0.0, "none", None)
        self.similarity_cache[cache_key] = result
        return result
    
    async def find_similar_values(self, data: pd.DataFrame, column: str) -> List[ValueGroup]:
        """Find groups of similar values in the dataset."""
        unique_values = {str(v) for v in data[column].dropna().unique() if v and not pd.isna(v)}
        self.metrics = ConsolidationMetrics(initial_unique_count=len(unique_values))
        
        processed = set()
        groups = []
        
        # First try LLM-based grouping if enabled
        if self.config.use_llm:
            try:
                llm_groups = await self._get_llm_groups(unique_values)
                groups.extend(llm_groups)
                processed.update(*(group.values for group in llm_groups))
            except Exception as e:
                print(f"LLM grouping failed, falling back to algorithmic approach: {str(e)}")
        
        # Fall back to algorithmic approach for remaining values
        remaining = unique_values - processed
        if remaining:
            algorithmic_groups = self._find_algorithmic_groups(remaining)
            self.metrics.algorithmic_groups_found = len(algorithmic_groups)
            groups.extend(algorithmic_groups)
        
        self.value_groups = groups
        self.metrics.total_groups_found = len(groups)
        
        return groups
    
    def _find_algorithmic_groups(self, values: Set[str]) -> List[ValueGroup]:
        """Find groups using algorithmic similarity."""
        processed = set()
        groups = []
        
        # First pass: Find exact matches and variations
        for val1 in sorted(values):
            if val1 in processed:
                continue
            
            current_group = {val1}
            variations1 = self._get_variations(val1)
            
            # Check against all other values
            for val2 in sorted(values):
                if val1 == val2 or val2 in processed:
                    continue
                
                # Check if any variations match
                variations2 = self._get_variations(val2)
                if variations1 & variations2:
                    current_group.add(val2)
                    continue
                
                # Check similarity
                similarity, match_type, explanation = self._calculate_similarity(val1, val2)
                if similarity >= self.config.patterns[0].threshold:
                    current_group.add(val2)
            
            if len(current_group) > 1:
                group = ValueGroup(current_group)
                group.match_explanation = "Text variations and abbreviations"
                groups.append(group)
                processed.update(current_group)
        
        return groups
    
    def get_consolidation_preview(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Preview the dataset with suggested consolidations."""
        if not self.value_groups:
            self.find_similar_values(data, column)
        
        # Create mapping dictionary
        value_mapping = {}
        for group in self.value_groups:
            for value in group.values:
                value_mapping[value] = group.canonical_form
        
        # Create preview
        preview = data.copy()
        preview[f"{column}_consolidated"] = preview[column].map(value_mapping).fillna(preview[column])
        
        return preview
    
    def apply_consolidation(
        self, 
        data: pd.DataFrame, 
        column: str, 
        group_overrides: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Apply consolidation with optional user-provided canonical forms."""
        if group_overrides:
            for canonical, group_values in group_overrides.items():
                matching_group = next(
                    (g for g in self.value_groups if g.values == set(group_values)), 
                    None
                )
                if matching_group:
                    matching_group.set_canonical_form(canonical)
        
        return self.get_consolidation_preview(data, column)
    
    def get_metrics(self) -> ConsolidationMetrics:
        """Return the current consolidation metrics."""
        return self.metrics
    
    def modify_group(self, group_index: int, action: str, value: Optional[str] = None, new_values: Optional[Set[str]] = None) -> bool:
        """Modify a group's values or structure.
        
        Args:
            group_index: Index of the group to modify
            action: One of 'remove_value', 'update_values', 'set_canonical'
            value: Value to remove or set as canonical
            new_values: New set of values for the group
        
        Returns:
            bool: Whether the modification was successful
        """
        if not 0 <= group_index < len(self.value_groups):
            return False
        
        group = self.value_groups[group_index]
        
        if action == "remove_value" and value:
            if not group.remove_value(value):
                # If group becomes too small, remove it
                self.value_groups.pop(group_index)
            return True
        
        elif action == "update_values" and new_values:
            if len(new_values) >= 2:
                group.update_values(new_values)
                return True
            else:
                # Remove group if too small
                self.value_groups.pop(group_index)
                return True
        
        elif action == "set_canonical" and value:
            try:
                if value in group.values:
                    group.set_canonical_form(value)
                else:
                    group.canonical_form = value
                return True
            except ValueError:
                return False
        
        return False
    
    def split_group(self, group_index: int, value: str) -> Optional[ValueGroup]:
        """Split a value from a group into its own group."""
        if not 0 <= group_index < len(self.value_groups):
            return None
        
        group = self.value_groups[group_index]
        if value not in group.values:
            return None
        
        # Remove the value from the original group
        if group.remove_value(value):
            # Create a new group with just this value
            new_group = ValueGroup({value})
            self.value_groups.append(new_group)
            return new_group
        
        return None
