from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def main():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Words to compare
    words = ["3 in 1 shampoo", "3 in 1 conditioner"]
    
    # Get embeddings
    embedded = await embeddings.aembed_documents(words)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedded[0], embedded[1])
    
    print(f"\nCosine similarity between '{words[0]}' and '{words[1]}': {similarity:.4f}")
    print(f"Distance: {1 - similarity:.4f}")

if __name__ == "__main__":
    load_dotenv()
    import asyncio
    asyncio.run(main()) 