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
conda create -n pureml_consolidation_agent python=3.10
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

