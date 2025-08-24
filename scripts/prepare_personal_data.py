#!/usr/bin/env python3
"""
Personal Data Preparation Script

Prepare personal food taste labels for training by mapping to HuggingFace Food101 dataset.

Author: Noam Shani
Course: 046211 Deep Learning, Technion

Attribution:
- pandas: Data manipulation and CSV handling
- scikit-learn: Stratified train/test splitting
- HuggingFace datasets: Food101 dataset access
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def load_food101_categories(categories_file: str = 'data/food101_categories.json') -> dict:
    """Load Food101 category ID to name mapping."""
    try:
        with open(categories_file, 'r') as f:
            categories = json.load(f)
        print(f"✅ Loaded {len(categories)} Food101 categories")
        return categories
    except FileNotFoundError:
        print(f"⚠️ Categories file not found: {categories_file}")
        print("Extracting categories from HuggingFace dataset...")
        
        # Load dataset to extract categories
        dataset = load_dataset("food101")
        category_names = dataset['train'].features['label'].names
        
        # Create mapping
        categories = {str(i): name for i, name in enumerate(category_names)}
        
        # Save for future use
        Path(categories_file).parent.mkdir(parents=True, exist_ok=True)
        with open(categories_file, 'w') as f:
            json.dump(categories, f, indent=2)
        
        print(f"✅ Extracted and saved {len(categories)} categories to {categories_file}")
        return categories

def map_personal_labels_to_hf(
    personal_labels_file: str,
    categories: dict,
    output_dir: str = 'data'
) -> pd.DataFrame:
    """
    Map personal labels to HuggingFace dataset indices.
    
    Args:
        personal_labels_file: Path to Excel/CSV file with personal labels
        categories: Food101 category mapping
        output_dir: Output directory for processed data
        
    Returns:
        DataFrame with mapped personal labels
    """
    print(f"📊 Loading personal labels from {personal_labels_file}")
    
    # Load personal labels
    if personal_labels_file.endswith('.xlsx'):
        df = pd.read_excel(personal_labels_file)
    else:
        df = pd.read_csv(personal_labels_file)
    
    print(f"Loaded {len(df)} personal labels")
    print(f"Columns: {list(df.columns)}")
    
    # Display label distribution
    if 'label' in df.columns:
        label_dist = df['label'].value_counts()
        print(f"Label distribution: {dict(label_dist)}")
    
    # Create reverse category mapping (name -> id)
    name_to_id = {name: int(id_str) for id_str, name in categories.items()}
    
    # Map food categories to HF dataset indices
    mapped_data = []
    unmapped_count = 0
    
    for idx, row in df.iterrows():
        # Extract food category from image path
        if 'image_path' in row:
            image_path = row['image_path']
            # Extract category name from path (e.g., "/content/food-101/train/pizza/123.jpg" -> "pizza")
            if '/food-101/' in image_path:
                category_name = image_path.split('/food-101/')[1].split('/')[1]  # train or validation
                food_category = image_path.split('/food-101/')[1].split('/')[2]  # food category
            else:
                # Try to extract from filename
                filename = Path(image_path).stem
                category_name = filename.split('_')[0] if '_' in filename else filename
                food_category = category_name
        else:
            # If no image_path, try food_category column
            food_category = row.get('food_category', 'unknown')
        
        # Map to HF dataset index
        if food_category in name_to_id:
            hf_category_id = name_to_id[food_category]
            
            # Map personal label to class ID
            personal_label = row.get('label', 'neutral').lower()
            label_mapping = {'disgusting': 0, 'neutral': 1, 'tasty': 2}
            class_id = label_mapping.get(personal_label, 1)  # Default to neutral
            
            mapped_data.append({
                'food_category': food_category,
                'hf_category_id': hf_category_id,
                'personal_label': personal_label,
                'class_id': class_id,
                'original_image_path': row.get('image_path', ''),
                'hf_split': 'train',  # Default to train split
                'hf_idx': idx  # This would need proper mapping to actual HF indices
            })
        else:
            unmapped_count += 1
            print(f"⚠️ Could not map food category: {food_category}")
    
    if unmapped_count > 0:
        print(f"⚠️ {unmapped_count} samples could not be mapped")
    
    # Create DataFrame
    mapped_df = pd.DataFrame(mapped_data)
    
    if len(mapped_df) > 0:
        print(f"✅ Successfully mapped {len(mapped_df)} samples")
        print(f"Class distribution: {dict(mapped_df['class_id'].value_counts().sort_index())}")
        
        # Save raw mapped data
        output_path = Path(output_dir) / 'raw_personal_labels.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mapped_df.to_csv(output_path, index=False)
        print(f"💾 Saved raw mapped data to {output_path}")
    
    return mapped_df

def create_stratified_splits(
    df: pd.DataFrame,
    split_strategy: str = '3way',
    random_seed: int = 42,
    output_dir: str = 'data'
) -> tuple:
    """
    Create stratified train/val/test splits.
    
    Args:
        df: DataFrame with mapped personal labels
        split_strategy: '2way' (80/20) or '3way' (60/20/20)
        random_seed: Random seed for reproducibility
        output_dir: Output directory for split files
        
    Returns:
        Tuple of (train_df, val_df, test_df) or (train_df, None, test_df) for 2way
    """
    print(f"🔄 Creating {split_strategy} stratified splits...")
    
    # Set random seed
    np.random.seed(random_seed)
    
    if split_strategy == '2way':
        # 80/20 split
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['class_id'], 
            random_state=random_seed
        )
        val_df = None
        
        print(f"📊 2-way split created:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        
        # Save splits
        output_dir = Path(output_dir)
        train_path = output_dir / 'personal_train_2way.csv'
        test_path = output_dir / 'personal_test_2way.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"💾 Saved splits:")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")
        
        return train_df, None, test_df
        
    else:  # 3way split
        # 60/20/20 split
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.4, 
            stratify=df['class_id'], 
            random_state=random_seed
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df['class_id'], 
            random_state=random_seed
        )
        
        print(f"📊 3-way split created:")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Val: {len(val_df)} samples")
        print(f"   Test: {len(test_df)} samples")
        
        # Save splits
        output_dir = Path(output_dir)
        train_path = output_dir / 'personal_train_3way.csv'
        val_path = output_dir / 'personal_val_3way.csv'
        test_path = output_dir / 'personal_test_3way.csv'
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"💾 Saved splits:")
        print(f"   Train: {train_path}")
        print(f"   Val: {val_path}")
        print(f"   Test: {test_path}")
        
        return train_df, val_df, test_df

def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description='Prepare personal food taste data for training')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to personal labels file (Excel or CSV)')
    parser.add_argument('--split', choices=['2way', '3way'], default='3way',
                       help='Split strategy: 2way (80/20) or 3way (60/20/20)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for processed data')
    parser.add_argument('--categories', type=str, default='data/food101_categories.json',
                       help='Path to Food101 categories mapping file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("🍽️ Personal Food Taste Data Preparation")
    print("=" * 50)
    print(f"Input file: {args.data}")
    print(f"Split strategy: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    try:
        # Load Food101 categories
        categories = load_food101_categories(args.categories)
        
        # Map personal labels to HF dataset
        mapped_df = map_personal_labels_to_hf(
            args.data,
            categories,
            args.output_dir
        )
        
        if len(mapped_df) == 0:
            print("❌ No samples could be mapped. Please check your data format.")
            return
        
        # Create stratified splits
        train_df, val_df, test_df = create_stratified_splits(
            mapped_df,
            args.split,
            args.seed,
            args.output_dir
        )
        
        # Print summary
        print(f"\n📋 Data Preparation Summary:")
        print(f"   Total mapped samples: {len(mapped_df)}")
        print(f"   Train samples: {len(train_df)}")
        if val_df is not None:
            print(f"   Validation samples: {len(val_df)}")
        print(f"   Test samples: {len(test_df)}")
        
        # Class distribution summary
        print(f"\n📊 Final Class Distribution:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if split_df is not None:
                dist = dict(split_df['class_id'].value_counts().sort_index())
                print(f"   {split_name}: {dist}")
        
        print(f"\n✅ Data preparation completed successfully!")
        print(f"📂 Files saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
