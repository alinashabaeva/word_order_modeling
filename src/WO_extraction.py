"""
Word Order Extraction Module

This module analyzes word order patterns in extracted transitive sentences
using Stanza dependency parsing. It identifies the positions of
Subject (S), Verb (V), and Object (O) in sentences (SVO, SOV, VSO, VOS, OSV, OVS).
"""

import pandas as pd
from pathlib import Path
from config import MAIN_DATA_FILE


def find_word_positions(sentence, subject_text, verb_text, object_text, nlp_stanza):
    """
    Find the positions of subject, verb, and object in the sentence
    Returns a tuple of (subject_pos, verb_pos, object_pos) or None if not found
    """
    doc = nlp_stanza(sentence)
    
    word_positions = {}
    for sent in doc.sentences:
        for i, word in enumerate(sent.words):
            if word.text not in word_positions:
                word_positions[word.text] = []
            word_positions[word.text].append(i)
    
    # Find positions for each constituent
    subject_pos = None
    verb_pos = None 
    object_pos = None
    
    # Find subject position
    if subject_text in word_positions:
        subject_pos = min(word_positions[subject_text])
    else:
        subject_words = subject_text.split()
        for word in subject_words:
            if word in word_positions:
                subject_pos = min(word_positions[word])
                break
    
    # Find verb position
    if verb_text in word_positions:
        verb_pos = min(word_positions[verb_text])
    else:
        # Try to find the first word of the verb constituent
        verb_words = verb_text.split()
        for word in verb_words:
            if word in word_positions:
                verb_pos = min(word_positions[word])
                break
    
    # Find object position
    if object_text in word_positions:
        object_pos = min(word_positions[object_text])
    else:
        # Try to find the first word of the object constituent
        object_words = object_text.split()
        for word in object_words:
            if word in word_positions:
                object_pos = min(word_positions[word])
                break
    
    if subject_pos is not None and verb_pos is not None and object_pos is not None:
        return (subject_pos, verb_pos, object_pos)
    else:
        return None

def determine_word_order(subject_pos, verb_pos, object_pos):
    """
    Determine word order based on positions of S, V, O
    Returns one of: SVO, SOV, VSO, VOS, OSV, OVS
    """
    positions = [
        ('S', subject_pos),
        ('V', verb_pos), 
        ('O', object_pos)
    ]
    
    positions.sort(key=lambda x: x[1]) # sort by position
     
    order = ''.join([pos[0] for pos in positions])
    return order

def analyze_word_order(sentence, subject_constituent, verb_constituent, object_constituent, nlp_stanza):
    """
    Analyze word order for a given sentence and its constituents
    Returns the word order pattern (SVO, SOV, etc.) or 'UNKNOWN' if analysis fails
    """
    try:
        positions = find_word_positions(
            sentence, 
            subject_constituent, 
            verb_constituent, 
            object_constituent,
            nlp_stanza
        )
        
        if positions:
            subject_pos, verb_pos, object_pos = positions
            word_order = determine_word_order(subject_pos, verb_pos, object_pos)
            return word_order
        else:
            return 'UNKNOWN'
            
    except Exception as e:
        print(f"Error analyzing word order for sentence: {str(e)[:50]}...")
        return 'UNKNOWN'

def add_word_order_column(csv_file_path, nlp_stanza):
    """
    Add word order column to existing CSV file with sentence extraction results
    """
    df = pd.read_csv(csv_file_path)
    
    print(f"Analyzing word order for {len(df)} sentences...")
    
    word_orders = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:  # update the progress every 100 sentences
            print(f"Processed {idx}/{len(df)} sentences for word order analysis")
            
        word_order = analyze_word_order(
            row['sentence'],
            row['subject_constituent'],
            row['verb_constituent'], 
            row['object_constituent'],
            nlp_stanza
        )
        word_orders.append(word_order)
    
    df['word_order'] = word_orders
    
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    print(f"Word order analysis complete! Updated {csv_file_path}")
    
    # word order statistics
    word_order_counts = df['word_order'].value_counts()
    print("\nOverall word order distribution:")
    for order, count in word_order_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {order}: {count} ({percentage:.1f}%)")
    
    # per-file word order statistics 
    if 'source_file' in df.columns:
        print("\nPer-file word order distribution:")
        for source_file in df['source_file'].unique():
            file_df = df[df['source_file'] == source_file]
            print(f"\n  {source_file} ({len(file_df)} sentences):")
            file_word_order_counts = file_df['word_order'].value_counts()
            for order, count in file_word_order_counts.items():
                percentage = (count / len(file_df)) * 100
                print(f"    {order}: {count} ({percentage:.1f}%)")
    
    return df

def main(nlp_stanza):
    df = add_word_order_column(MAIN_DATA_FILE, nlp_stanza)
    return df
