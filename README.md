### PureML Consolidation Agent

This is a Python script that uses the PureML Consolidation Agent to consolidate similar values in a CSV file.

Overview: 
- In NewRAGAgent.py, we introduce a different approach to consolidation. 
- We extract unique values from the dataset, embed + store them in a vector database. 
- Perform a similarity search on the vector database to find similar values considering mispellings, abbreviations, synonyms, bad formatting, etc. 
- Form groups of similar values and present them to the user for consolidation decisions (e.g. user can edit groupings and resolve them manually)

To run the script: 
- Create a .env file with your OpenAI API key
- Then create and activate a conda environment:
```
conda create -n pureml_consolidation_agent python=3.12
conda activate pureml_consolidation_agent
```
- Install the requirements:
```
pip install -r requirements.txt
```
- Run the script:
```
python test_rag_agent.py
```
- In test_rag_agent.py, you can test different datasets by changing the path to the dataset in the script. 

Some Thoughts:
- Current implementation might be a bit computationally expensive for bigger datasets where there's many many unique values. Should explore some sort of optimization technique whether it be batching, some sort of parallelization, etc.
- Accuracy will likely improve if we inject some context about the dataset into the prompt as well. These can easily be extracted from user's initial inputs in the PureML UI. Specifically the LLM should have some context on what the column represents/role it plays in the dataset (e.g. in a dataset about cars, the brand column refers to the brand of the car)
- Prompt + LLM should also be configurable by user.
- Will add a feature where we suggest a canonical value based on the value that appears the most in the dataset in that particular mapping.
- Needs to be thoroughly tested.
