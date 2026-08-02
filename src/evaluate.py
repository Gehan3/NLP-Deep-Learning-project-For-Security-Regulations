import json
import pandas as pd
import numpy as np
import os
from pathlib import Path

from retriever import ISO27002Retriever 
#from lexical_representation import LexicalRetriever
retriever = ISO27002Retriever()

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "..", "data", "Ground_truth.json")

with open(json_path, "r", encoding="utf-8") as f:
    ground_truth_data = json.load(f)

def evaluate_retrieval(retriever, ground_truth_data, k=3):
    precisions, recalls, hit_rates, reciprocal_ranks = [], [], [], []
    
    for item in ground_truth_data:
        query_text = item["scenario"]
        
        raw_expected = item["expected_controls"].split("|")
        relevant_controls = set()
        for c in raw_expected:
            parts = c.strip().split()
            if parts:
                ctrl_id = parts[0].strip("—").strip("-").strip()
                relevant_controls.add(ctrl_id)
        
        _, sources = retriever.build_context(query_text, k=k+2, max_sources=k)
        #update remove duplicate
        raw_retrieved = [str(source["control_id"]).strip() for source in sources]
        retrieved_controls = []
        for ctrl in raw_retrieved:
            if ctrl not in retrieved_controls:
                retrieved_controls.append(ctrl)
        
        retrieved_controls = [str(source["control_id"]).strip() for source in sources]
        print(f"Scenario: {query_text[:40]}...")
        print(f"Expected Controls : {relevant_controls}")
        print(f"Retrieved Controls: {retrieved_controls}")
        print("-" * 40)

        
        #Metrics
        # Hit Rate: 
        hit = 1 if any(ctrl in relevant_controls for ctrl in retrieved_controls) else 0
        hit_rates.append(hit)
        
        # Precision@K
        retrieved_relevant_count = sum(1 for ctrl in retrieved_controls if ctrl in relevant_controls)
        precision = retrieved_relevant_count / k if k > 0 else 0
        precisions.append(precision)
        
        # Recall@K
        recall = retrieved_relevant_count / len(relevant_controls) if len(relevant_controls) > 0 else 0.0
        recalls.append(recall)
        
        # MRR (Mean Reciprocal Rank)
        rr = 0.0
        for rank, ctrl in enumerate(retrieved_controls, start=1):
            if ctrl in relevant_controls:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
    evaluation_results = {
        f"Precision@{k}": np.mean(precisions),
        f"Recall@{k}": np.mean(recalls),
        f"Hit Rate@{k}": np.mean(hit_rates),
        "MRR": np.mean(reciprocal_ranks)
    }
    
    return pd.DataFrame([evaluation_results])

if __name__ == "__main__":
   
    retriever = ISO27002Retriever()
    #retriever = LexicalRetriever()
    

    evaluation_df = evaluate_retrieval(retriever, ground_truth_data, k=3)
    print("\n--- Evaluation Results ---")
    print(evaluation_df)