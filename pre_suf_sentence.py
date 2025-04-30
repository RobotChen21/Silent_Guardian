import json
import os
import torch
import numpy as np
from tqdm.auto import tqdm
import argparse
import sys
import datetime
import re

# Import functions from the provided scripts
from json_filter import filter_json_by_max_prob


def extract_model_name(model_path):
    """
    Extract the model name from the model path.
    Example:
    - From "TheBloke/vicuna-13b-v1.5" extracts "vicuna"
    - From "/path/to/TheBloke/vicuna-13b-v1.5" extracts "vicuna"

    :param model_path: Path to the model
    :return: Extracted model name
    """
    try:
        # Get the last part of the path (in case there are directories)
        base_name = os.path.basename(model_path)
        if not base_name:  # If basename is empty (e.g., path ends with /)
            base_name = os.path.basename(os.path.dirname(model_path))

        # If there's a slash in the path (like TheBloke/vicuna-13b)
        if '/' in model_path:
            parts = model_path.split('/')
            for part in parts:
                if '-' in part:
                    base_name = part
                    break

        # Extract the part before the first hyphen
        match = re.search(r'([^-/]+)', base_name)
        if match:
            return match.group(1).lower()

        # Fallback: if no hyphen found, use the basename without extension
        return os.path.splitext(base_name)[0].lower()
    except Exception as e:
        print(f"Error extracting model name: {e}, using 'model' as default")
        return "model"


def load_prefixes_suffixes(prefix_suffix_path):
    """
    Load prefixes and suffixes from a JSON file.

    File format: [
        "prefix1", "prefix1_chinese",
        "prefix2", "prefix2_chinese",
        "prefix3", "prefix3_chinese",
        "suffix1", "suffix1_chinese",
        "suffix2", "suffix2_chinese",
        "suffix3", "suffix3_chinese"
    ]

    :param prefix_suffix_path: Path to the JSON file containing prefixes and suffixes
    :return: Tuple of (english_prefixes, chinese_prefixes, english_suffixes, chinese_suffixes)
    """
    try:
        with open(prefix_suffix_path, 'r', encoding='utf-8') as f:
            prefix_suffix_data = json.load(f)

        if len(prefix_suffix_data) != 12:
            print(f"Warning: Expected 12 items in prefix/suffix file, got {len(prefix_suffix_data)}")

        # Extract English and Chinese versions
        english_prefixes = [prefix_suffix_data[i] for i in range(0, 6, 2)]
        chinese_prefixes = [prefix_suffix_data[i] for i in range(1, 6, 2)]

        english_suffixes = [prefix_suffix_data[i] for i in range(6, 12, 2)]
        chinese_suffixes = [prefix_suffix_data[i] for i in range(7, 12, 2)]

        print(f"Loaded {len(english_prefixes)} English prefixes and {len(chinese_prefixes)} Chinese prefixes")
        print(f"Loaded {len(english_suffixes)} English suffixes and {len(chinese_suffixes)} Chinese suffixes")

        return english_prefixes, chinese_prefixes, english_suffixes, chinese_suffixes

    except Exception as e:
        print(f"Error loading prefix/suffix file: {str(e)}")
        sys.exit(1)


def identify_text_field(sentences):
    """
    Identify the field that contains the text content in the sentences.

    :param sentences: List of sentence items from JSON
    :return: Name of the text field
    """
    # Common field names for text content
    possible_text_fields = ['text', 'content', 'sentence', 'prompt', 'message', 'adv', 'origin']

    if not sentences or not isinstance(sentences[0], dict):
        print("Warning: Sentences data is empty or not in expected format")
        return 'adv'  # Default to adv field

    # Check which field exists in the first sentence
    sample = sentences[0]
    print(f"Sample sentence structure: {list(sample.keys())}")

    # 优先检查 adv 字段
    if 'adv' in sample:
        print(f"Using 'adv' as the text field")
        return 'adv'

    # 如果没有 adv 字段，则检查其他可能字段
    for field in possible_text_fields:
        if field in sample:
            print(f"Using '{field}' as the text field")
            return field

    # If no known field is found, use the field with the longest string value
    longest_field = None
    longest_length = 0

    for key, value in sample.items():
        if isinstance(value, str) and len(value) > longest_length:
            longest_field = key
            longest_length = len(value)

    if longest_field:
        print(f"Using '{longest_field}' as the text field (longest string value)")
        return longest_field

    print("Warning: Could not identify text field, defaulting to 'adv'")
    return 'adv'


def categorize_sentences(sentences, text_field):
    """
    Organize sentences into the 6 specified categories.

    :param sentences: List of sentence items from JSON
    :param text_field: Name of the field containing the text content
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
        "english_120tokens_2": [],
        "chinese_120tokens_2": []
    }

    # First 5 are English 40 tokens
    categories["english_40tokens"] = sentences[:5]
    # Next 5 are Chinese 40 tokens
    categories["chinese_40tokens"] = sentences[5:10]
    # Next 5 are English 80 tokens (group 1)
    categories["english_80tokens_1"] = sentences[10:15]
    # Next 5 are Chinese 80 tokens (group 1)
    categories["chinese_80tokens_1"] = sentences[15:20]
    # Next 5 are English 120 tokens (group 2)
    categories["english_120tokens_2"] = sentences[20:25]
    # Last 5 are Chinese 120 tokens (group 2)
    categories["chinese_120tokens_2"] = sentences[25:30]

    return categories


def is_chinese_category(category_name):
    """
    Determine if a category is Chinese based on its name.

    :param category_name: Name of the category
    :return: True if it's a Chinese category, False otherwise
    """
    return "chinese" in category_name


def create_combined_sentences(categories, eng_prefixes, cn_prefixes, eng_suffixes, cn_suffixes, text_field):
    """
    Combine sentences with either a prefix or a suffix (not both).
    - English sentences with English prefixes/suffixes
    - Chinese sentences with Chinese prefixes/suffixes

    :param categories: Dictionary of categorized sentences
    :param eng_prefixes: List of English prefix strings
    :param cn_prefixes: List of Chinese prefix strings
    :param eng_suffixes: List of English suffix strings
    :param cn_suffixes: List of Chinese suffix strings
    :param text_field: Name of the field containing the text content (应该是 'adv')
    :return: Dictionary mapping classification to combined sentences
    """
    classifications = {}

    # 正则表达式模式用于去除 <xxxxx> 格式的标签
    tag_pattern = re.compile(r'<[^>]+>')

    # For each category and each prefix or suffix
    for cat_name, sentences in categories.items():
        # Determine if this is a Chinese category
        is_chinese = is_chinese_category(cat_name)

        # Select the appropriate prefixes and suffixes based on language
        prefixes = cn_prefixes if is_chinese else eng_prefixes
        suffixes = cn_suffixes if is_chinese else eng_suffixes

        # Process prefixes (prefix + sentence)
        for prefix_idx, prefix in enumerate(prefixes):
            classification = f"{cat_name}_prefix_{prefix_idx}"
            classifications[classification] = []

            # Combine each sentence in this category with this prefix
            for sentence in sentences:
                # 获取 adv 字段的内容，如果不存在则尝试使用 text_field 字段
                # 注意：确保这里使用 adv 字段而不是 origin
                raw_text = sentence.get("adv", sentence.get(text_field, ""))

                # 如果 adv 和 text_field 都不存在，尝试获取 origin
                if not raw_text and "origin" in sentence:
                    raw_text = sentence["origin"]
                    print(f"Warning: Using 'origin' field as fallback for sentence text")

                # 去除 <xxxxx> 标签
                sentence_text = tag_pattern.sub('', raw_text).strip()

                combined = {
                    "origin": sentence.get("origin", ""),  # 保留原始 origin 用于引用
                    "text": f"{prefix} {sentence_text}",  # 使用前缀组合
                    "raw_adv": raw_text,  # 保存原始未处理的 adv 内容
                    "adv": sentence_text,  # 保存处理后的 adv 内容
                    "category": cat_name,
                    "prefix_index": prefix_idx,
                    "suffix_index": -1,  # No suffix
                    "language": "chinese" if is_chinese else "english",
                    "orig_prob": sentence.get("prob", 0)
                }
                classifications[classification].append(combined)

        # Process suffixes (sentence + suffix)
        for suffix_idx, suffix in enumerate(suffixes):
            classification = f"{cat_name}_suffix_{suffix_idx}"
            classifications[classification] = []

            # Combine each sentence in this category with this suffix
            for sentence in sentences:
                # 获取 adv 字段的内容，如果不存在则尝试使用 text_field 字段
                # 注意：确保这里使用 adv 字段而不是 origin
                raw_text = sentence.get("adv", sentence.get(text_field, ""))

                # 如果 adv 和 text_field 都不存在，尝试获取 origin
                if not raw_text and "origin" in sentence:
                    raw_text = sentence["origin"]
                    print(f"Warning: Using 'origin' field as fallback for sentence text")

                # 去除 <xxxxx> 标签
                sentence_text = tag_pattern.sub('', raw_text).strip()

                combined = {
                    "origin": sentence.get("origin", ""),  # 保留原始 origin 用于引用
                    "text": f"{sentence_text} {suffix}",  # 使用后缀组合
                    "raw_adv": raw_text,  # 保存原始未处理的 adv 内容
                    "adv": sentence_text,  # 保存处理后的 adv 内容
                    "category": cat_name,
                    "prefix_index": -1,  # No prefix
                    "suffix_index": suffix_idx,
                    "language": "chinese" if is_chinese else "english",
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
    try:
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
    except Exception as e:
        print(f"Error calculating probability for text: {e}")
        return 0.0


def process_classifications(classifications, model, tokenizer, device):
    """
    Process each classification, calculating probabilities for each sentence
    and computing the average.

    :param classifications: Dictionary mapping classification to sentences
    :param model: The language model
    :param tokenizer: The tokenizer for the model
    :param device: Device to run the model on
    :return: Dictionary of classification averages and summary data
    """
    results = {}
    summary = {}

    for classification, sentences in tqdm(classifications.items(), desc="Processing classifications"):
        probs = []
        orig_probs = []

        for sentence in tqdm(sentences, desc=f"Processing {classification}", leave=False):
            prob = calculate_probability(model, tokenizer, sentence["text"], device)
            sentence["prob"] = prob
            probs.append(prob)
            orig_probs.append(sentence["orig_prob"])

        # Calculate average probability for this classification
        avg_prob = sum(probs) / len(probs) if probs else 0
        avg_orig_prob = sum(orig_probs) / len(orig_probs) if orig_probs else 0

        # Store detailed results
        results[classification] = {
            "sentences": sentences,
            "average_prob": avg_prob,
            "average_orig_prob": avg_orig_prob
        }

        # Add to summary for easier analysis
        parts = classification.split('_')
        category = parts[0] + "_" + parts[1]  # e.g., english_40tokens
        if len(parts) > 3:  # For cases like english_80tokens_1_prefix_0
            category = parts[0] + "_" + parts[1] + "_" + parts[2]

        affix_type = parts[-2]  # prefix or suffix
        affix_idx = int(parts[-1])  # index number

        if category not in summary:
            summary[category] = {
                "prefixes": {},
                "suffixes": {}
            }

        if affix_type == "prefix":
            summary[category]["prefixes"][affix_idx] = {
                "avg_prob": avg_prob,
                "avg_orig_prob": avg_orig_prob
            }
        else:  # suffix
            summary[category]["suffixes"][affix_idx] = {
                "avg_prob": avg_prob,
                "avg_orig_prob": avg_orig_prob
            }

    return results, summary


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
        model = AutoGPTQForCausalLM.from_quantized(
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
    parser = argparse.ArgumentParser(description="Process sentences with prefixes or suffixes using a quantized model")
    parser.add_argument("input_file", help="Path to the input JSON file with sentences")
    parser.add_argument("prefix_suffix_file", help="Path to the JSON file with prefixes and suffixes")
    parser.add_argument("--model_path", required=True, help="Path to the quantized model")
    parser.add_argument("--device", default="cuda", help="Device to run the model on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--text_field", default="adv", help="Field name for text content")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = "pre_suf"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Extract model name from model path
    model_name = extract_model_name(args.model_path)
    print(f"Extracted model name: {model_name}")

    # Generate output filename based on date and model name
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    output_file = f"{output_dir}/{model_name}_{date_str}.json"
    summary_file = f"{output_dir}/{model_name}_{date_str}_summary.json"

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

    # 2. Identify the field containing text content (默认使用 adv 字段)
    text_field = args.text_field if args.text_field else identify_text_field(filtered_sentences)

    # 3. Load prefixes and suffixes (now with English and Chinese versions)
    eng_prefixes, cn_prefixes, eng_suffixes, cn_suffixes = load_prefixes_suffixes(args.prefix_suffix_file)

    # 4. Categorize sentences
    categories = categorize_sentences(filtered_sentences, text_field)

    # 5. Create combined sentences for each classification (only prefix OR suffix, not both)
    classifications = create_combined_sentences(
        categories,
        eng_prefixes, cn_prefixes,
        eng_suffixes, cn_suffixes,
        text_field
    )

    # 6. Load quantized model and tokenizer
    model, tokenizer = load_quantized_model_and_tokenizer(args.model_path, device)

    # 7. Process classifications and calculate probabilities
    results, summary = process_classifications(classifications, model, tokenizer, device)

    # 8. Save detailed results to JSON
    print(f"Saving detailed results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # 9. Save summary results to JSON
    print(f"Saving summary results to {summary_file}...")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"Processing complete. Results saved to {output_file} and {summary_file}")

    # 10. Print summary of average probabilities
    print("\nSummary of average probabilities by category:")
    for category, data in summary.items():
        print(f"\n{category}:")

        print("  Prefixes:")
        for idx, values in data["prefixes"].items():
            print(f"    Prefix {idx}: avg_prob={values['avg_prob']:.6f}, avg_orig_prob={values['avg_orig_prob']:.6f}")

        print("  Suffixes:")
        for idx, values in data["suffixes"].items():
            print(f"    Suffix {idx}: avg_prob={values['avg_prob']:.6f}, avg_orig_prob={values['avg_orig_prob']:.6f}")


if __name__ == "__main__":
    main()