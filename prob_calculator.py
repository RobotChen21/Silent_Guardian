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


def get_file_type(input_file):
    """
    Determine if the input file is a target or novel type

    :param input_file: Input file path
    :return: 'target' or 'novel'
    """
    filename = os.path.basename(input_file).lower()
    if 'target' in filename:
        return 'target'
    elif 'novel' in filename:
        return 'novel'
    else:
        # Default to target if can't determine
        print(f"Warning: Could not determine file type from name '{filename}'. Defaulting to 'target'.")
        return 'target'


def generate_output_filename(input_file, model_path, file_type):
    """
    Generate output filename based on input file and model path
    Format: model1->model2_target/novel_yyyyMMdd.json
    """
    # Extract the model1 part from input filename (model1_dataset_time.json)
    input_basename = os.path.basename(input_file)
    model1_match = re.match(r'([^_]+)_.*\.json', input_basename)
    model1_part = model1_match.group(1) if model1_match else "model"

    # Extract model2 from model path (TheBloke/model2_xxxx)
    model_path_parts = os.path.normpath(model_path).split(os.path.sep)
    if len(model_path_parts) >= 2:
        model2_full = model_path_parts[-1]  # Get the last part after the slash
        model2_match = re.match(r'([^-]+)', model2_full)
        model2_part = model2_match.group(1) if model2_match else model2_full
    else:
        model2_part = os.path.basename(os.path.normpath(model_path))

    # Get current date in format YYYYMMDD
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Create output filename with the new format model1->model2_target_yyyyMMdd.json
    return f"{model1_part}->{model2_part}_{file_type}_{current_date}.json"


def categorize_sentences(file_type):
    """
    Return the mapping of categories to their expected order in filtered data
    based on whether it's a target or novel file type
    """
    if file_type == 'target':
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
    elif file_type == 'novel':
        categories = {
            "en_40tokens": (0, 5),  # 5 English 40 tokens (0-4)
            "zh_40tokens": (5, 10),  # 5 Chinese 40 tokens (5-9)
            "en_80tokens": (10, 15),  # 5 English 80 tokens (10-14)
            "zh_80tokens": (15, 20),  # 5 Chinese 80 tokens (15-19)
            "en_120tokens": (20, 25),  # 5 English 120 tokens (20-24)
            "zh_120tokens": (25, 30)  # 5 Chinese 120 tokens (25-29)
        }
    else:
        # Should never reach here, but just in case
        categories = {}

    return categories


def main():
    parser = argparse.ArgumentParser(description="Filter JSON data and calculate end token probabilities")
    parser.add_argument("--path", type=str, required=True, help="Path to the model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file with original data")
    args = parser.parse_args()

    # Determine file type (target or novel)
    file_type = get_file_type(args.input_file)
    print(f"Processing file as type: {file_type}")

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

    # Get category mapping based on file type
    categories = categorize_sentences(file_type)

    # Initialize category-based data containers for both original and new probs
    original_category_probs = defaultdict(list)
    new_category_probs = defaultdict(list)
    results = []

    # Process each sentence
    for i, target in enumerate(tqdm(targets)):
        tokens = torch.tensor(tokenizer.encode(target)).to(device)

        # Calculate new probability
        new_prob = calculate_probability(model, tokenizer, tokens)

        # Get original probability from filtered data
        original_prob = filtered_data[i].get('prob', 0)

        # Store results, including original data
        result = filtered_data[i].copy()  # Copy all original fields
        result["model_prob"] = new_prob  # Add new probability as a separate field
        results.append(result)

        # Determine category based on position
        category_found = False
        for category_name, (start_idx, end_idx) in categories.items():
            if start_idx <= i < end_idx:
                original_category_probs[category_name].append(original_prob)
                new_category_probs[category_name].append(new_prob)
                result["category"] = category_name  # Add category to the result
                category_found = True
                break

        if not category_found:
            print(f"Warning: Could not categorize sentence at index {i}")
            result["category"] = "unknown"

        # Print results
        print(f"Sentence: {target}")
        print(f"Category: {result['category']}")
        print(f"Original Probability: {original_prob}")
        print(f"New Model Probability: {new_prob}")
        print("==============")

        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate average probabilities for each category (both original and new)
    category_comparisons = {}
    for category in set(list(original_category_probs.keys()) + list(new_category_probs.keys())):
        original_probs = original_category_probs.get(category, [])
        new_probs = new_category_probs.get(category, [])

        original_avg = sum(original_probs) / len(original_probs) if original_probs else 0
        new_avg = sum(new_probs) / len(new_probs) if new_probs else 0

        category_comparisons[category] = {
            "original_avg_prob": original_avg,
            "new_avg_prob": new_avg,
            "difference": new_avg - original_avg
        }

        print(f"Category: {category}")
        print(f"  Original Average Probability: {original_avg}")
        print(f"  New Average Probability: {new_avg}")
        print(f"  Difference: {new_avg - original_avg}")

    # Create a final results structure
    final_results = {
        "individual_results": results,
        "category_comparisons": category_comparisons
    }

    # Create output directory if it doesn't exist
    output_dir = 'probability_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate dynamic output filename with appropriate type
    result_name = generate_output_filename(args.input_file, args.path, file_type)

    # Save results to JSON file
    json_path = os.path.join(output_dir, result_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, indent=4, ensure_ascii=False, fp=f)

    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()