# Consolidation Algorithm

This package provides a robust, dataset-agnostic approach to consolidating similar values in datasets using string similarity metrics and LLM verification.

## Features

- Dataset agnostic - works with any type of textual data
- Parallel processing for improved performance
- String similarity using Jaro-Winkler distance
- LLM-based verification of groups
- Smart subgroup detection with certainty-based grouping
- Per-dataset cost tracking and detailed logging
- Interactive graph visualization
- Value limit protection (max 1,000 unique values per column)

## Installation

```bash
pip install -r requirements.txt
```

Make sure to set your OpenAI API key in your environment:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Configuration Parameters

- `similarity_threshold`: Threshold for Jaro-Winkler string similarity (default: 0.8)
- `model`: The OpenAI model to use for verification (default: "gpt-4")
- `batch_size`: Number of concurrent LLM calls (default: 5)
- `MAX_UNIQUE_VALUES`: Maximum number of unique values to process per column (default: 1,000)

## Testing

To test the consolidation algorithm with predefined datasets and configurations:

```bash
python test_bench.py
```

The test bench:
1. Processes multiple test datasets in sequence
2. Generates detailed results for each dataset including:
   - Number of rows processed
   - Number of unique values
   - Groups identified
   - Cost metrics
3. Saves results in JSON format in the `results/` directory with format:
   - `{dataset_name}_{num_rows}rows_{timestamp}.json`
4. Allows for reproducible testing and performance comparison

### Test Results Structure

Each test result JSON includes:
- Dataset information (name, total rows, sampling)
- Column-specific results (groups found, similarity scores)
- Cost metrics (input tokens, output tokens, total cost)
- Processing timestamp

This structured testing approach helps in:
- Validating algorithm performance
- Tracking costs across different datasets
- Comparing results between algorithm versions
- Identifying potential improvements

## Algorithm Steps

1. Column Analysis
   - Generate concise column description using LLM
   - Extract unique values (limited to first 1,000)
   - Build column context

2. Parallel Processing
   - Standardize values by removing special characters
   - Process values in efficient batches

3. Initial Grouping
   - Form initial groups from exact standardized matches
   - Build similarity graph using Jaro-Winkler distance
   - Find connected components

4. LLM Verification
   - Verify groups in parallel batches
   - Analyze rejected groups for valid subgroups
   - Only form subgroups with 100% certainty
   - Place values in best-fit groups when overlaps occur

5. Interactive Review
   - Visualize similarity graph
   - Review and select canonical forms
   - Apply verified mappings to dataset

6. Cost Tracking
   - Track LLM input/output tokens per dataset
   - Calculate API costs per dataset
   - Display cost summary after processing

## Output

The algorithm produces:
1. Consolidated CSV file with mapped values
2. Similarity graph visualization
3. Detailed cost metrics per dataset
4. Processing logs with group information

## Notes

- For columns with more than 1,000 unique values, only the first 1,000 values will be processed
- The algorithm prioritizes certainty in grouping over completeness
- Cost metrics are tracked separately for each dataset
- Graph visualization can be saved with custom filenames

## Sujan's Additional Notes
- In testing, it's pretty clear that we need some more filtering of groups than just Jaro-Winkler similarity. If the LLM is analyzing really large groups, that's blows up the context window which makes things expensive or can cause the LLM to fail.
- This filtering can be done with bolstering our similarity calculations with more sophisticated methods as I mentioned previously or even by switching over to a 3rd partyVector DB in the long-run. 
- This is a good starting point for the algorithm with regards to MVP. 


My Notes: 
- Embeddings aren't really being used at all for the similarity calculation. In my testing, I've noticed that cosine similarity b/w embeddings isn't as great of a measure as Jaro Winkler Distance for our use cases so far. Jaro Winkler is good at catching the bad formatting/mispellings and the cosine similarity calculation is a little better for synonyms. Couldn't really establish a clear pattern here, but I'm thinking in the future we could incorporate some sort of hybrid similarity calculation using embeddings and algorithms like Jaro Winkler (there's also other string similarity algorithms we can explore). For MVP, we can take out all the embeddings functionality (cost savings is super minimal, it'll just speed things up more). I left it in for now. 
- If we decide to use embeddings later on, choosing the right Vector DB would be really helpful for our use case. DBs like Pinecone, Qdrant, etc. come with lots of neat functionality that we can use to find/cluster embeddings that are similar. We could abandon/rely less on string similarity algorithms if this avenue works. Could probably switch to some sort of purely vector searching method. 

