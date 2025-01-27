# Consolidation Algorithm

This package provides a robust, dataset-agnostic approach to consolidating similar values in datasets using semantic embeddings, string similarity metrics, and LLM verification.

## Features

- Dataset agnostic - works with any type of textual data
- Parallel processing for improved performance
- Semantic similarity using OpenAI embeddings
- String similarity using Jaro-Winkler distance
- LLM-based verification of groups
- Automatic subgroup detection for rejected groups
- Caching of embeddings for efficiency
- Detailed cost tracking and logging
- Interactive graph visualization

## Installation

```bash
pip install -r requirements.txt
```

Make sure to set your OpenAI API key in your environment:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

Simply run the script and follow the interactive prompts:

```bash
python IntelligentConsolidation.py
```

The script will:
1. Ask for your CSV file path
2. Show available columns and their unique value counts
3. Let you select which column to analyze
4. Process the column and find similar value groups
5. Allow you to review each group and choose canonical forms
6. Show cost summary
7. Let you save the consolidated results to a new CSV

## Configuration Parameters

- `similarity_threshold`: Threshold for Jaro-Winkler string similarity (default: 0.8)
- `model`: The OpenAI model to use for verification (default: "gpt-4")
- `embedding_model`: The OpenAI embedding model (default: "text-embedding-3-small")
- `batch_size`: Number of concurrent LLM calls (default: 5)

## Algorithm Steps

1. Column Analysis
   - Generate concise column description using LLM
   - Extract unique values
   - Build column context

2. Parallel Processing
   - Generate embeddings for unique values
   - Standardize values by removing special characters

3. Initial Grouping
   - Form initial groups from exact standardized matches
   - Build similarity graph using Jaro-Winkler distance
   - Find connected components

4. LLM Verification
   - Verify groups in parallel batches
   - Analyze rejected groups for valid subgroups
   - Calculate embedding variance for visualization

5. Interactive Review
   - Visualize similarity graph
   - Review and select canonical forms
   - Apply verified mappings to dataset

6. Cost Tracking
   - Monitor embedding token usage
   - Track LLM input/output tokens
   - Calculate total API costs

My Notes: 
- Embeddings aren't really being used at all for the similarity calculation. In my testing, I've noticed that cosine similarity b/w embeddings isn't as great of a measure as Jaro Winkler Distance for our use cases so far. Jaro Winkler is good at catching the bad formatting/mispellings and the cosine similarity calculation is a little better for synonyms. Couldn't really establish a clear pattern here, but I'm thinking in the future we could incorporate some sort of hybrid similarity calculation using embeddings and algorithms like Jaro Winkler (there's also other string similarity algorithms we can explore). For MVP, we can take out all the embeddings functionality (cost savings is super minimal, it'll just speed things up more). I left it in for now. 
- If we decide to use embeddings later on, choosing the right Vector DB would be really helpful for our use case. DBs like Pinecone, Qdrant, etc. come with lots of neat functionality that we can use to find/cluster embeddings that are similar. We could abandon/rely less on string similarity algorithms if this avenue works. Could probably switch to some sort of purely vector searching method. 

