import json
import os
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import sys

# Import functions from the provided scripts
from json_filter import filter_json_by_max_prob


def load_prefixes_suffixes(prefix_suffix_path):
    """
    Load prefixes and suffixes from a JSON file.

    :param prefix_suffix_path: Path to the JSON file containing prefixes and suffixes
    :return: Tuple of (prefixes, suffixes)
    """
    try:
        with open(prefix_suffix_path, 'r', encoding='utf-8') as f:
            prefix_suffix_data = json.load(f)

        if len(prefix_suffix_data) != 6:
            print(f"Warning: Expected 6 items in prefix/suffix file, got {len(prefix_suffix_data)}")

        prefixes = prefix_suffix_data[:3]
        suffixes = prefix_suffix_data[3:]

        print(f"Loaded {len(prefixes)} prefixes and {len(suffixes)} suffixes")
        return prefixes, suffixes
    except Exception as e:
        print(f"Error loading prefix/suffix file: {str(e)}")
        sys.exit(1)


def categorize_sentences(sentences):
    """
    Organize sentences into the 6 specified categories.

    :param sentences: List of sentence items from JSON
    :return: Dictionary mapping category to list of sentences
    """
    # Verify we have 30 sentences
    if len(sentences) != 30:
        print(f"Warning: Expected 30 sentences, got {len(sentences)}")

    # Create categories based on description
    categories = {
        "english_40tokens": [],
        "chinese_40tokens": [],
        "english_80tokens_1": [],
        "chinese_80tokens_1": [],
        "english_80tokens_2": [],
        "chinese_80tokens_2": []
    }

    # First 5 are English 40 tokens
    categories["english_40tokens"] = sentences[:5]
    # Next 5 are Chinese 40 tokens
    categories["chinese_40tokens"] = sentences[5:10]
    # Next 5 are English 80 tokens (group 1)
    categories["english_80tokens_1"] = sentences[10:15]
    # Next 5 are Chinese 80 tokens (group 1)
    categories["chinese_80tokens_1"] = sentences[15:20]
    # Next 5 are English 80 tokens (group 2)
    categories["english_80tokens_2"] = sentences[20:25]
    # Last 5 are Chinese 80 tokens (group 2)
    categories["chinese_80tokens_2"] = sentences[25:30]

    return categories


def create_combined_sentences(categories, prefixes, suffixes):
    """
    Combine each sentence with each prefix and suffix.

    :param categories: Dictionary of categorized sentences
    :param prefixes: List of prefix strings
    :param suffixes: List of suffix strings
    :return: Dictionary mapping classification to combined sentences
    """
    classifications = {}

    # For each category and each prefix and suffix
    for cat_name, sentences in categories.items():
        for prefix in prefixes:
            for suffix in suffixes:
                classification = f"{cat_name}_{prefixes.index(prefix)}_{suffixes.index(suffix)}"
                classifications[classification] = []

                # Combine each sentence in this category with this prefix and suffix
                for sentence in sentences:
                    combined = {
                        "origin": sentence["origin"],
                        "text": f"{prefix} {sentence['text']} {suffix}",
                        "category": cat_name,
                        "prefix_index": prefixes.index(prefix),
                        "suffix_index": suffixes.index(suffix),
                        "orig_prob": sentence.get("prob", 0)
                    }
                    classifications[classification].append(combined)

    print(f"Created {len(classifications)} classifications with combined sentences")
    return classifications


def calculate_probability(model, tokenizer, text, device):
    """
    Calculate the probability for a given text using the method from STP.py.

    :param model: The language model
    :param tokenizer: The tokenizer for the model
    :param text: The text to compute probability for
    :param device: Device to run the model on
    :return: Probability value
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    init = inputs.input_ids[0]

    # Similar to what's in step() function in STP.py
    with torch.no_grad():
        logits = model(init.unsqueeze(0)).logits
        eos_token_id = tokenizer.eos_token_id
        # Get probability of EOS token
        prob = torch.nn.functional.softmax(logits[0, -1, :], dim=0)[eos_token_id]

    return prob.item()


def process_classifications(classifications, model, tokenizer, device):
    """
    Process each classification, calculating probabilities for each sentence
    and computing the average.

    :param classifications: Dictionary mapping classification to sentences
    :param model: The language model
    :param tokenizer: The tokenizer for the model
    :param device: Device to run the model on
    :return: Dictionary of classification averages
    """
    results = {}

    for classification, sentences in tqdm(classifications.items(), desc="Processing classifications"):
        probs = []

        for sentence in tqdm(sentences, desc=f"Processing {classification}", leave=False):
            prob = calculate_probability(model, tokenizer, sentence["text"], device)
            sentence["prob"] = prob
            probs.append(prob)

        # Calculate average probability for this classification
        avg_prob = sum(probs) / len(probs) if probs else 0
        results[classification] = {
            "sentences": sentences,
            "average_prob": avg_prob
        }

    return results


def load_quantized_model_and_tokenizer(model_path, device):
    """
    Load a quantized model using AutoGPTQForCausalLM.

    :param model_path: Path to the quantized model
    :param device: Device to load the model onto
    :return: Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM

        print(f"Loading quantized model from {model_path} to {device}")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Load the quantized model
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantize_config=None
        ).to(device)

        print("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading quantized model: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Process sentences with prefixes and suffixes using a quantized model")
    parser.add_argument("input_file", help="Path to the input JSON file with sentences")
    parser.add_argument("prefix_suffix_file", help="Path to the JSON file with prefixes and suffixes")
    parser.add_argument("--model_path", required=True, help="Path to the quantized model")
    parser.add_argument("--device", default="cuda", help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--output", default="classification_results.json", help="Output file path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    # 1. Process the input file using json_filter
    print("Filtering JSON by max probability...")
    filtered_sentences, _ = filter_json_by_max_prob(args.input_file)

    if not filtered_sentences:
        print("Failed to process input file.")
        sys.exit(1)

    # 2. Load prefixes and suffixes
    prefixes, suffixes = load_prefixes_suffixes(args.prefix_suffix_file)

    # 3. Categorize sentences
    categories = categorize_sentences(filtered_sentences)

    # 4. Create combined sentences for each classification
    classifications = create_combined_sentences(categories, prefixes, suffixes)

    # 5. Load quantized model and tokenizer
    model, tokenizer = load_quantized_model_and_tokenizer(args.model_path, device)

    # 6. Process classifications and calculate probabilities
    results = process_classifications(classifications, model, tokenizer, device)

    # 7. Save results to JSON
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. Results saved to {args.output}")

    # 8. Print average probabilities for each classification
    print("\nAverage probabilities by classification:")
    for classification, data in results.items():
        print(f"{classification}: {data['average_prob']:.6f}")


if __name__ == "__main__":
    main()