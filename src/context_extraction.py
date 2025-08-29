"""
Context Extraction Module

This module extracts context for transitive sentences by finding:
1. 2 preceding speech sentences (column D)
3. Skips camera (column E) and name (column G) annotations

Context structure:
- 2 preceding speech sentences
- Comments (column F) between target and first preceding speech
- Comments between the 2 preceding speech sentences  
- Only closest comment (1 sentence) before the first speech sentence)
"""

import os
import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from config import MAIN_DATA_FILE

def find_tsv_files(data_folder):
    """Find all TSV files in the data folder"""
    tsv_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.tsv'):
                filepath = os.path.join(root, file)
                tsv_files.append((filepath, file))
    return tsv_files

def load_tsv_data(filepath, filename):
    """Load TSV data and organize by sentence"""
    sentence_data = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip() or line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            
            # Skip lines that don't have enough columns (need 7 for A-G)
            if len(parts) < 7:
                continue

            token_id, _, word, column_d, column_e, column_f, column_g = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6] # Column A, B, C, D(=speech), E(=camera), F(=comment), G(=name)
                
            
            if '-' in token_id:
                try:
                    sentence_num, token_num = token_id.split('-', 1)
                    token_pos = int(token_num)
                    
                    token_info = {
                        'token_id': token_id,
                        'word': word,
                        'speech': column_d,
                        'camera': column_e,
                        'comment': column_f,
                        'name': column_g,
                        'token_pos': token_pos,
                        'line_num': line_num,
                        'sentence_num': int(sentence_num)
                    }
                    
                    sentence_key = f"{filename}_{sentence_num}"
                    sentence_data[sentence_key].append(token_info)
                    
                except (ValueError, IndexError):
                    continue
    
    return sentence_data

def reconstruct_sentence_by_annotation(tokens):
    """
    Reconstruct sentence parts by annotation type, similar to sent_extraction.py
    Returns dict with 'speech', 'comment', 'camera', 'name' parts
    """
    tokens.sort(key=lambda x: x['token_pos'])
    
    # Separate words by annotation type
    speech_words = []
    comment_words = []
    camera_words = []
    name_words = []
    
    for token in tokens:
        if token['word']:
            # Check each annotation type (a token can have multiple)
            if token['speech'].strip() and token['speech'] != '_':
                speech_words.append(token['word'])
            if token['comment'].strip() and token['comment'] != '_':
                comment_words.append(token['word'])
            if token['camera'].strip() and token['camera'] != '_':
                camera_words.append(token['word'])
            if token['name'].strip() and token['name'] != '_':
                name_words.append(token['word'])
    
    # Reconstruct text for each annotation type
    result = {}
    for ann_type, words in [('speech', speech_words), ('comment', comment_words), 
                           ('camera', camera_words), ('name', name_words)]:
        if words:
            sentence_text = ' '.join(words)
            sentence_text = re.sub(r'\s+([.,!?;:])', r'\1', sentence_text)
            result[ann_type] = sentence_text
        else:
            result[ann_type] = ""
    
    return result

def should_include_sentence(sentence_parts):
    """
    Determine if a sentence should be included in context
    Skip if it only has camera/name annotations, include if it has speech/comment
    """
    has_speech = bool(sentence_parts['speech'].strip())
    has_comment = bool(sentence_parts['comment'].strip())
    has_camera = bool(sentence_parts['camera'].strip())
    has_name = bool(sentence_parts['name'].strip())
    
    # Include if it has speech or comment, skip if it only has camera/name
    if has_speech or has_comment:
        return True
    elif has_camera or has_name:
        return False
    else:
        return False  # Skip empty sentences

def extract_context_for_sentence(target_sentence_key, all_sentence_data):
    """
    Extract context for a target transitive sentence
    1. Find target sentence in raw data
    2. Find 2 closest preceding SPEECH sentences (skip camera/name)
    3. Include comments between last speech and target
    4. Include comments between the 2 speech sentences
    5. Include 1 adjacent comment before the first (furthest) speech sentence (if present)
    """
    target_tokens = all_sentence_data.get(target_sentence_key, [])
    if not target_tokens:
        return ""
    
    target_sentence_num = target_tokens[0]['sentence_num']
    filename = target_sentence_key.split('_')[0]
    
    all_preceding_sentences = []
    for sent_key, tokens in all_sentence_data.items():
        if sent_key.startswith(filename + '_'):
            sentence_num = tokens[0]['sentence_num']
            if sentence_num < target_sentence_num:
                sentence_parts = reconstruct_sentence_by_annotation(tokens)
                
                all_preceding_sentences.append({
                    'sentence_num': sentence_num,
                    'sentence_parts': sentence_parts
                })
    
    all_preceding_sentences.sort(key=lambda x: x['sentence_num'])  # sort by sentence number (ascending - furthest to closest)
    
    # STEP 1: Find the 2 closest sentences with SPEECH content
    speech_sentences = []
    for sent in reversed(all_preceding_sentences):  # start from closest to target
        if sent['sentence_parts']['speech'].strip():
            speech_sentences.append(sent)
            if len(speech_sentences) == 2:
                break
    
    if len(speech_sentences) < 2:
        return ""  # Need at least 2 preceding speech sentences
    
    # speech_sentences[0] = closest to target, speech_sentences[1] = furthest from target
    closest_speech = speech_sentences[0]
    furthest_speech = speech_sentences[1]
    
    context_parts = []
    
    # STEP 2: Find comment that is adjacent (immediately before) the furthest speech sentence
    for sent in all_preceding_sentences:
        if sent['sentence_num'] == furthest_speech['sentence_num'] - 1:
            if sent['sentence_parts']['comment'].strip():
                context_parts.append(f"COMMENT: {sent['sentence_parts']['comment']}")
            break
    
    # STEP 3: Add the furthest speech sentence (with mixed annotations if present)
    furthest_parts = furthest_speech['sentence_parts']
    if furthest_parts['comment'].strip():
        context_parts.append(f"COMMENT: {furthest_parts['comment']}")
    if furthest_parts['speech'].strip():
        context_parts.append(f"SPEECH: {furthest_parts['speech']}")
    
    # STEP 4: Include ALL sentences between the 2 speech sentences (with mixed annotations)
    for sent in all_preceding_sentences:
        if furthest_speech['sentence_num'] < sent['sentence_num'] < closest_speech['sentence_num']:
            parts = sent['sentence_parts']
            if parts['comment'].strip():
                context_parts.append(f"COMMENT: {parts['comment']}")
            if parts['speech'].strip():
                context_parts.append(f"SPEECH: {parts['speech']}")
    
    # STEP 5: Add the closest speech sentence (with mixed annotations if present)
    closest_parts = closest_speech['sentence_parts']
    if closest_parts['comment'].strip():
        context_parts.append(f"COMMENT: {closest_parts['comment']}")
    if closest_parts['speech'].strip():
        context_parts.append(f"SPEECH: {closest_parts['speech']}")
    
    # STEP 6: Include ALL sentences between closest speech and target (with mixed annotations)
    for sent in all_preceding_sentences:
        if closest_speech['sentence_num'] < sent['sentence_num'] < target_sentence_num:
            parts = sent['sentence_parts']
            if parts['comment'].strip():
                context_parts.append(f"COMMENT: {parts['comment']}")
            if parts['speech'].strip():
                context_parts.append(f"SPEECH: {parts['speech']}")
    
    return '\n'.join(context_parts)

def add_context_to_results(results_csv_path):
    """
    Add context column to existing transitive sentence results
    """
    df = pd.read_csv(results_csv_path)
    
    data_folder = Path('/Users/AlyaMac/Desktop/STONY BROOK/YEAR 2/QP/code/word_order_Russian/data/raw')
    tsv_files = find_tsv_files(data_folder)
    
    all_sentence_data = {}
    for filepath, filename in tsv_files:
        sentence_data = load_tsv_data(filepath, filename)
        all_sentence_data.update(sentence_data)
        
    contexts = []
    successful_extractions = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        sentence_id = row['id']
        
        if '_verb' in sentence_id:
            base_id = sentence_id.split('_verb')[0]
        else:
            base_id = sentence_id
            
        context = extract_context_for_sentence(base_id, all_sentence_data)
        contexts.append(context)
        
        if context:
            successful_extractions += 1
    
    df['context'] = contexts
    
    df.to_csv(results_csv_path, index=False, encoding='utf-8')
    
    return df

def main():
    df = add_context_to_results(MAIN_DATA_FILE)
    return df


