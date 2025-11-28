"""
Token Length Statistics for Subtask 1 Dataset

This script analyzes token lengths after tokenization with bert-base-multilingual-cased
for each language-domain combination in the training and validation datasets.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict
import pandas as pd


def load_jsonl(filepath: str) -> List[Dict]:
    """
    Load JSONL file.
    
    Args:
        filepath: path to JSONL file
    
    Returns:
        list of dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_aspect_text_pairs(data: List[Dict]) -> List[str]:
    """
    Extract aspect-text pairs from JSONL data, following the same format as VADataset.
    
    Args:
        data: list of dictionaries from JSONL file
    
    Returns:
        list of formatted strings "aspect: text"
    """
    pairs = []
    
    for item in data:
        text = item.get("Text", "")
        
        # Handle different formats
        if 'Quadruplet' in item:
            for quad in item['Quadruplet']:
                aspect = quad.get('Aspect', '')
                pairs.append(f"{aspect}: {text}")
        elif 'Triplet' in item:
            for triplet in item['Triplet']:
                aspect = triplet.get('Aspect', '')
                pairs.append(f"{aspect}: {text}")
        elif 'Aspect_VA' in item:
            for aspect_va in item['Aspect_VA']:
                aspect = aspect_va.get('Aspect', '')
                pairs.append(f"{aspect}: {text}")
        elif 'Aspect' in item:
            for aspect in item['Aspect']:
                pairs.append(f"{aspect}: {text}")
    
    return pairs


def compute_token_lengths(texts: List[str], tokenizer) -> List[int]:
    """
    Compute token lengths for a list of texts.
    
    Args:
        texts: list of text strings
        tokenizer: HuggingFace tokenizer
    
    Returns:
        list of token lengths
    """
    lengths = []
    for text in texts:
        tokens = tokenizer(text, truncation=False, add_special_tokens=True)
        lengths.append(len(tokens['input_ids']))
        if len(lengths)==1:
            print(text,tokens,lengths)

    return lengths


def plot_distribution(train_lengths: List[int], dev_lengths: List[int], 
                     lang: str, domain: str, output_dir: Path):
    """
    Plot token length distribution for training and validation sets.
    
    Args:
        train_lengths: token lengths for training set
        dev_lengths: token lengths for validation set
        lang: language code
        domain: domain name
        output_dir: output directory path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms with density=True to show frequency (normalized)
    bins = np.arange(0, max(max(train_lengths), max(dev_lengths)) + 10, 5)
    
    # Use density=True to calculate frequency, and weights to normalize
    ax.hist(train_lengths, bins=bins, alpha=0.6, label='Training Set', 
            color='blue', edgecolor='darkblue', linewidth=1.2, 
            histtype='stepfilled', density=True)
    ax.hist(dev_lengths, bins=bins, alpha=0.6, label='Validation Set', 
            color='red', edgecolor='darkred', linewidth=1.2, 
            histtype='stepfilled', density=True)
    
    # Add statistics
    train_mean = np.mean(train_lengths)
    train_median = np.median(train_lengths)
    train_max = np.max(train_lengths)
    dev_mean = np.mean(dev_lengths)
    dev_median = np.median(dev_lengths)
    dev_max = np.max(dev_lengths)
    
    # Add vertical lines for means
    ax.axvline(train_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Train Mean: {train_mean:.1f}')
    ax.axvline(dev_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Val Mean: {dev_mean:.1f}')
    
    ax.set_xlabel('Token Length', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Token Length Distribution - {lang.upper()} {domain.capitalize()}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (f'Train: n={len(train_lengths)}, mean={train_mean:.1f}, median={train_median:.1f}, max={train_max}\n'
                 f'Val: n={len(dev_lengths)}, mean={dev_mean:.1f}, median={dev_median:.1f}, max={dev_max}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"{lang}_{domain}_token_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_overall_distribution(all_train_lengths: List[int], all_dev_lengths: List[int], 
                             output_dir: Path):
    """
    Plot overall token length distribution for all datasets combined.
    
    Args:
        all_train_lengths: token lengths for all training sets
        all_dev_lengths: token lengths for all validation sets
        output_dir: output directory path
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot histograms with density=True to show frequency (normalized)
    bins = np.arange(0, max(max(all_train_lengths), max(all_dev_lengths)) + 10, 5)
    
    # Use density=True to calculate frequency
    ax.hist(all_train_lengths, bins=bins, alpha=0.6, label='Training Set', 
            color='blue', edgecolor='darkblue', linewidth=1.2, 
            histtype='stepfilled', density=True)
    ax.hist(all_dev_lengths, bins=bins, alpha=0.6, label='Validation Set', 
            color='red', edgecolor='darkred', linewidth=1.2, 
            histtype='stepfilled', density=True)
    
    # Add statistics
    train_mean = np.mean(all_train_lengths)
    train_median = np.median(all_train_lengths)
    train_max = np.max(all_train_lengths)
    dev_mean = np.mean(all_dev_lengths)
    dev_median = np.median(all_dev_lengths)
    dev_max = np.max(all_dev_lengths)
    
    # Add vertical lines for means
    ax.axvline(train_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8, label=f'Train Mean: {train_mean:.1f}')
    ax.axvline(dev_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Val Mean: {dev_mean:.1f}')
    
    ax.set_xlabel('Token Length', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Token Length Distribution - All Datasets Combined', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (f'Train: n={len(all_train_lengths)}, mean={train_mean:.1f}, median={train_median:.1f}, max={train_max}\n'
                 f'Val: n={len(all_dev_lengths)}, mean={dev_mean:.1f}, median={dev_median:.1f}, max={dev_max}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "overall_token_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def save_statistics_summary(stats_dict: Dict, output_dir: Path):
    """
    Save detailed statistics to JSON and text files.
    
    Args:
        stats_dict: dictionary containing statistics for each dataset
        output_dir: output directory path
    """
    # Save as JSON
    json_path = output_dir / "token_length_statistics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")
    
    # Save as formatted text
    txt_path = output_dir / "token_length_statistics.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TOKEN LENGTH STATISTICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for key, stats in stats_dict.items():
            if key == 'overall':
                f.write("\n" + "=" * 80 + "\n")
                f.write("OVERALL STATISTICS (All Datasets Combined)\n")
                f.write("=" * 80 + "\n")
            else:
                lang, domain = key.split('_', 1)
                f.write(f"\n{lang.upper()} - {domain.capitalize()}\n")
                f.write("-" * 80 + "\n")
            
            for split in ['train', 'dev']:
                if split in stats:
                    f.write(f"  {split.capitalize()}:\n")
                    f.write(f"    Count: {stats[split]['count']}\n")
                    f.write(f"    Mean: {stats[split]['mean']:.2f}\n")
                    f.write(f"    Median: {stats[split]['median']:.2f}\n")
                    f.write(f"    Std: {stats[split]['std']:.2f}\n")
                    f.write(f"    Min: {stats[split]['min']}\n")
                    f.write(f"    Max: {stats[split]['max']}\n")
                    f.write(f"    25th percentile: {stats[split]['q25']:.2f}\n")
                    f.write(f"    75th percentile: {stats[split]['q75']:.2f}\n")
                    f.write(f"    95th percentile: {stats[split]['q95']:.2f}\n")
                    f.write("\n")
    
    print(f"Saved: {txt_path}")


def compute_statistics(lengths: List[int]) -> Dict:
    """
    Compute comprehensive statistics for token lengths.
    
    Args:
        lengths: list of token lengths
    
    Returns:
        dictionary of statistics
    """
    return {
        'count': len(lengths),
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'q25': float(np.percentile(lengths, 25)),
        'q75': float(np.percentile(lengths, 75)),
        'q95': float(np.percentile(lengths, 95))
    }


def main():
    """
    Main function to process all datasets and generate statistics.
    """
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw_track_a" / "subtask_1"
    output_dir = project_root / "script" / "token_length_output"
    output_dir.mkdir(exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer: bert-base-multilingual-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}\n")
    
    # Define language-domain combinations
    lang_domains = [
        ('eng', 'laptop'),
        ('eng', 'restaurant'),
        ('jpn', 'finance'),
        ('jpn', 'hotel'),
        ('rus', 'restaurant'),
        ('tat', 'restaurant'),
        ('ukr', 'restaurant'),
        ('zho', 'finance'),
        ('zho', 'laptop'),
        ('zho', 'restaurant')
    ]
    
    # Storage for overall statistics
    all_train_lengths = []
    all_dev_lengths = []
    stats_dict = {}
    
    # Process each language-domain combination
    for lang, domain in lang_domains:
        print(f"Processing: {lang.upper()} - {domain.capitalize()}")
        
        # Determine file patterns
        lang_dir = data_dir / lang
        
        # Find train and dev files
        train_files = list(lang_dir.glob(f"{lang}_{domain}_train*.jsonl"))
        dev_files = list(lang_dir.glob(f"{lang}_{domain}_dev*.jsonl"))
        
        if not train_files or not dev_files:
            print(f"  Warning: Missing files for {lang}-{domain}")
            continue
        
        train_file = train_files[0]
        dev_file = dev_files[0]
        
        print(f"  Train file: {train_file.name}")
        print(f"  Dev file: {dev_file.name}")
        
        # Load data
        train_data = load_jsonl(train_file)
        dev_data = load_jsonl(dev_file)
        
        # Extract aspect-text pairs
        train_texts = extract_aspect_text_pairs(train_data)
        dev_texts = extract_aspect_text_pairs(dev_data)
        
        print(f"  Train samples: {len(train_texts)}")
        print(f"  Dev samples: {len(dev_texts)}")
        
        # Compute token lengths
        train_lengths = compute_token_lengths(train_texts, tokenizer)
        dev_lengths = compute_token_lengths(dev_texts, tokenizer)
        
    #     # Compute statistics
    #     train_stats = compute_statistics(train_lengths)
    #     dev_stats = compute_statistics(dev_lengths)
        
    #     print(f"  Train: mean={train_stats['mean']:.1f}, median={train_stats['median']:.1f}, max={train_stats['max']}")
    #     print(f"  Dev: mean={dev_stats['mean']:.1f}, median={dev_stats['median']:.1f}, max={dev_stats['max']}")
        
    #     # Store statistics
    #     stats_dict[f"{lang}_{domain}"] = {
    #         'train': train_stats,
    #         'dev': dev_stats
    #     }
        
    #     # Plot distribution
    #     plot_distribution(train_lengths, dev_lengths, lang, domain, output_dir)
        
    #     # Add to overall statistics
    #     all_train_lengths.extend(train_lengths)
    #     all_dev_lengths.extend(dev_lengths)
        
    #     print()
    
    # # Plot overall distribution
    # print("Creating overall distribution plot...")
    # plot_overall_distribution(all_train_lengths, all_dev_lengths, output_dir)
    
    # # Add overall statistics
    # stats_dict['overall'] = {
    #     'train': compute_statistics(all_train_lengths),
    #     'dev': compute_statistics(all_dev_lengths)
    # }
    
    # # Save statistics summary
    # print("\nSaving statistics summary...")
    # save_statistics_summary(stats_dict, output_dir)
    
    # print(f"\nAll done! Results saved to: {output_dir}")
    # print(f"Total plots generated: {len(lang_domains) + 1}")


if __name__ == "__main__":
    main()

