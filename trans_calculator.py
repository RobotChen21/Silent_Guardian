import json
import torch
import gc
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import os
import argparse
import sys
import datetime
import re
from collections import defaultdict


def filter_json_by_max_prob(input_file):
    """
    Filter JSON file, keeping only the item with the highest prob value for each origin

    :param input_file: Input JSON file path
    :return: Processed data
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist!")
            return None

        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create a dictionary to store the item with max prob for each origin
        origin_max_prob = {}

        # Find the max prob item for each origin
        for item in data:
            origin = item['origin']
            prob = item['prob']

            # If this origin hasn't been recorded yet, or current prob is higher than recorded prob
            if origin not in origin_max_prob or prob > origin_max_prob[origin]['prob']:
                origin_max_prob[origin] = item

        # Convert results back to a list
        result = list(origin_max_prob.values())

        print(f"Filtering complete: selected {len(result)} items from {len(data)} items")

        return result

    except json.JSONDecodeError:
        print(f"Error: '{input_file}' is not a valid JSON file!")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None


def calculate_probability(model, tokenizer, tokens):
    """Calculate the probability of the end token given the input tokens."""
    eos_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        logits = model(tokens.unsqueeze(0)).logits
        # Get probabilities for the last position
        probs = torch.nn.functional.softmax(logits[0, -1, :], dim=0)
        # For DeepSeek models, EOS token ID is 100001
        end_token_prob = float(probs[eos_token_id])
    return end_token_prob


def generate_output_filename(input_file, model_path):
    """
    Generate output filename based on input file and model path
    Format: model1_modelname_20250427.json
    """
    # Extract the model1 part from input filename (model1_dataset_time.json)
    input_basename = os.path.basename(input_file)
    model1_match = re.match(r'([^_]+)_.*\.json', input_basename)
    model1_part = model1_match.group(1) if model1_match else "model"

    # Extract model name from model path
    model_name = os.path.basename(os.path.normpath(model_path))

    # Get current date in format YYYYMMDD
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Create output filename
    return f"{model1_part}_{model_name}_{current_date}.json"


def categorize_sentences():
    """
    Return the mapping of categories to their expected order in filtered data
    """
    categories = {
        "写作": (0, 10),  # 10 writing sentences (0-9)
        "扮演": (10, 20),  # 10 roleplay sentences (10-19)
        "常识": (20, 30),  # 10 commonsense sentences (20-29)
        "物理": (30, 40),  # 10 physics sentences (30-39)
        "反事实": (40, 50),  # 10 counterfactual sentences (40-49)
        "编程": (50, 57),  # 7 programming sentences (50-56)
        "数学": (57, 60),  # 3 math sentences (57-59)
        "通用": (60, 70),  # 10 general sentences (60-69)
        "知识": (70, 80),  # 10 knowledge sentences (70-79)
        "违规问": (80, 90)  # 10 violation questions (80-89)
    }
    return categories


def main():
    parser = argparse.ArgumentParser(description="Filter JSON data and calculate end token probabilities")
    parser.add_argument("--path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file with original data")
    args = parser.parse_args()

    # First, filter the input JSON file
    filtered_data = filter_json_by_max_prob(args.input_file)
    if filtered_data is None:
        print("Failed to filter input data. Exiting.")
        sys.exit(1)

    # Extract sentences to evaluate from the filtered data
    targets = [item.get('sentence', item.get('origin', '')) for item in filtered_data]

    device = "cuda:0"

    # Load tokenizer and model
    tokenizer_path = model_path = args.path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantize_config=None
    ).to(device)

    # Get category mapping
    categories = categorize_sentences()

    # Initialize category-based data containers
    category_probs = defaultdict(list)
    results = []

    # Process each sentence
    for i, target in enumerate(tqdm(targets)):
        tokens = torch.tensor(tokenizer.encode(target)).to(device)

        # Calculate probability
        prob = calculate_probability(model, tokenizer, tokens)

        # Store results, including original data
        result = filtered_data[i].copy()  # Copy all original fields
        result["model_prob"] = prob  # Add new probability as a separate field
        results.append(result)

        # Determine category based on position
        category_found = False
        for category_name, (start_idx, end_idx) in categories.items():
            if start_idx <= i < end_idx:
                category_probs[category_name].append(prob)
                result["category"] = category_name  # Add category to the result
                category_found = True
                break

        if not category_found:
            print(f"Warning: Could not categorize sentence at index {i}")
            result["category"] = "unknown"

        # Print results
        print(f"Sentence: {target}")
        print(f"Category: {result['category']}")
        print(f"Model Probability: {prob}")
        print("==============")

        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate average probabilities for each category
    category_averages = {}
    for category, probs in category_probs.items():
        if probs:
            avg_prob = sum(probs) / len(probs)
            category_averages[category] = avg_prob
            print(f"Category: {category}, Average Probability: {avg_prob}")

    # Create a final results structure with both individual sentences and category averages
    final_results = {
        "individual_results": results,
        "category_averages": category_averages
    }

    # Create output directory if it doesn't exist
    output_dir = 'probability_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate dynamic output filename
    result_name = generate_output_filename(args.input_file, args.path)

    # Save results to JSON file
    json_path = os.path.join(output_dir, result_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, indent=4, ensure_ascii=False, fp=f)

    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()