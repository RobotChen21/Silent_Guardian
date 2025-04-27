import json
import os
import argparse
import numpy as np
from collections import defaultdict


def is_chinese(text):
    """Check if the text contains Chinese characters"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def calculate_edit_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def cosine_similarity(s1, s2):
    """Simple character-based cosine similarity"""
    # Create character frequency dictionaries
    s1_chars = {}
    s2_chars = {}

    for c in s1:
        s1_chars[c] = s1_chars.get(c, 0) + 1

    for c in s2:
        s2_chars[c] = s2_chars.get(c, 0) + 1

    # Find all unique characters
    all_chars = set(s1_chars.keys()).union(set(s2_chars.keys()))

    # Calculate dot product and magnitudes
    dot_product = 0
    magnitude_s1 = 0
    magnitude_s2 = 0

    for c in all_chars:
        dot_product += s1_chars.get(c, 0) * s2_chars.get(c, 0)
        magnitude_s1 += s1_chars.get(c, 0) ** 2
        magnitude_s2 += s2_chars.get(c, 0) ** 2

    magnitude_s1 = magnitude_s1 ** 0.5
    magnitude_s2 = magnitude_s2 ** 0.5

    if magnitude_s1 == 0 or magnitude_s2 == 0:
        return 0

    return dot_product / (magnitude_s1 * magnitude_s2)


def process_json_file(file_path):
    """Process a JSON file and calculate metrics by epoch"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Containers for English and Chinese data
    english_epochs = defaultdict(lambda: {'loss': [], 'prob': [], 'similarity': [], 'edit': []})
    chinese_epochs = defaultdict(lambda: {'loss': [], 'prob': [], 'similarity': [], 'edit': []})

    # Detect sentence pattern
    sentences = set()
    for item in data:
        if 'origin' in item:
            sentences.add(item['origin'])

    # If we couldn't identify the sentences, assume they're grouped by 30s
    if len(sentences) != 10:
        sentence_indices = {}
        for i, item in enumerate(data):
            if 'origin' in item:
                sentence_idx = i // 30
                sentence_indices[item['origin']] = sentence_idx
    else:
        # Each unique sentence gets 30 entries
        sentence_indices = {sentence: i for i, sentence in enumerate(sentences)}

    # Process each item
    for idx, item in enumerate(data):
        if 'origin' not in item or 'adv' not in item or 'loss' not in item or 'prob' not in item:
            continue

        origin = item['origin']
        adv = item['adv']
        loss = item['loss']
        prob = item['prob']

        # Remove the token if it exists
        if "<｜begin▁of▁sentence｜>" in adv:
            adv = adv.replace("<｜begin▁of▁sentence｜>", "")

        # Calculate similarity and edit distance
        similarity = cosine_similarity(origin, adv)
        edit = calculate_edit_distance(origin, adv)

        # Determine epoch (0-29)
        sentence_idx = sentence_indices.get(origin, idx // 30)
        epoch = idx % 30

        # Store in appropriate container
        if is_chinese(origin):
            chinese_epochs[epoch]['loss'].append(loss)
            chinese_epochs[epoch]['prob'].append(prob)
            chinese_epochs[epoch]['similarity'].append(similarity)
            chinese_epochs[epoch]['edit'].append(edit)
        else:
            english_epochs[epoch]['loss'].append(loss)
            english_epochs[epoch]['prob'].append(prob)
            english_epochs[epoch]['similarity'].append(similarity)
            english_epochs[epoch]['edit'].append(edit)

    # Calculate averages
    en_results = []
    cn_results = []

    for epoch in range(30):
        if epoch in english_epochs:
            en_results.append({
                'epoch': epoch + 1,
                'avg_loss': np.mean(english_epochs[epoch]['loss']),
                'avg_prob': np.mean(english_epochs[epoch]['prob']),
                'avg_similarity': np.mean(english_epochs[epoch]['similarity']),
                'avg_edit': np.mean(english_epochs[epoch]['edit'])
            })

        if epoch in chinese_epochs:
            cn_results.append({
                'epoch': epoch + 1,
                'avg_loss': np.mean(chinese_epochs[epoch]['loss']),
                'avg_prob': np.mean(chinese_epochs[epoch]['prob']),
                'avg_similarity': np.mean(chinese_epochs[epoch]['similarity']),
                'avg_edit': np.mean(chinese_epochs[epoch]['edit'])
            })

    return {
        'english': en_results,
        'chinese': cn_results
    }


def save_results(results, file_path, output_dir="painter_results"):
    """Save results to the specified output directory"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.basename(file_path).split('.')[0]

    # Save English results
    en_output_path = os.path.join(output_dir, f"{base_name}_english_results.json")
    with open(en_output_path, 'w', encoding='utf-8') as f:
        json.dump(results['english'], f, indent=2, ensure_ascii=False)

    # Save Chinese results
    cn_output_path = os.path.join(output_dir, f"{base_name}_chinese_results.json")
    with open(cn_output_path, 'w', encoding='utf-8') as f:
        json.dump(results['chinese'], f, indent=2, ensure_ascii=False)

    print(f"Results saved to {en_output_path} and {cn_output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process JSON files and calculate metrics by epoch')
    parser.add_argument('files', nargs='+', help='JSON files to process')
    parser.add_argument('--output-dir', default='painter_results', help='Output directory for results')

    args = parser.parse_args()

    for file_path in args.files:
        print(f"Processing {file_path}...")
        results = process_json_file(file_path)

        if results:
            save_results(results, file_path, args.output_dir)

            # Print summary
            print("\nSummary:")
            print("English sentences:")
            for i, epoch_data in enumerate(results['english']):
                if i < 3 or i >= len(results['english']) - 3:  # Show first 3 and last 3 epochs
                    print(f"  Epoch {epoch_data['epoch']}: Loss={epoch_data['avg_loss']:.4f}, "
                          f"Prob={epoch_data['avg_prob']:.4f}, "
                          f"Similarity={epoch_data['avg_similarity']:.4f}, "
                          f"Edit={epoch_data['avg_edit']:.2f}")
                elif i == 3:
                    print("  ...")

            print("\nChinese sentences:")
            for i, epoch_data in enumerate(results['chinese']):
                if i < 3 or i >= len(results['chinese']) - 3:  # Show first 3 and last 3 epochs
                    print(f"  Epoch {epoch_data['epoch']}: Loss={epoch_data['avg_loss']:.4f}, "
                          f"Prob={epoch_data['avg_prob']:.4f}, "
                          f"Similarity={epoch_data['avg_similarity']:.4f}, "
                          f"Edit={epoch_data['avg_edit']:.2f}")
                elif i == 3:
                    print("  ...")


if __name__ == "__main__":
    main()