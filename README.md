### PureML Consolidation Agent

This is a Python script that uses the PureML Consolidation Agent to consolidate similar values in a CSV file.

Some Thoughts: 
- Opening up the prompt to the user to hardcode in some concepts pertaining to the data would be helpful. 
- Getting the most accurate groupings relies on the LLM's ability to understand where the inconsistencies are. 
- There may be cases where we just need the LLM to change only value that's mispelled or poorly formatted. Maybe it could be useful to start with correcting those, then forming groups from there. 
- The reality is that a dataset might have similar entities that are not actually the same, it's beneficial to keep those entries as separate groups. An LLM may not be able to determine that, so it's important to have the user provide some context and maybe establish a human in the loop for group formation processing IN ADDITION to human in the loop for the final consolidation. 
- Currently everything is implemented in Pydantic (out of ease really), but can investigate LangGraph which may provide more support for really large datasets with their built-in parallelization strategies. 