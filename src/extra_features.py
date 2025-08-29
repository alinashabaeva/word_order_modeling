"""
Extra Features Extraction Module

This module adds linguistic features to the transitive sentence dataset:
1. POS tags for subject and object (using SpaCy)
2. Constituent size (count tokens) for subject, object, and verb
3. Syllable weight (count vowels) for subject, object, and verb constituents
4. Verb aspect analysis (using SpaCy)
5. Preceding question detection in context (if the last speech sentence in context ends with '?')
6. Negation presence in verb (using SpaCy) (if the verb is preceded by 'не' or 'ни')
7. Clause type (matrix/embedded) analysis (using SpaCy) 
8. NER for subject and object heads (using Stanza)
"""

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from config import MAIN_DATA_FILE

def get_pos_tags(subject_lemma, object_lemma, nlp_spacy):
    """
    Get POS tags for subject and object lemmas using SpaCy
    """
    try:
        subject_doc = nlp_spacy(subject_lemma)
        object_doc = nlp_spacy(object_lemma)
        
        subject_pos = subject_doc[0].pos_ if len(subject_doc) > 0 else "UNKNOWN"
        object_pos = object_doc[0].pos_ if len(object_doc) > 0 else "UNKNOWN"
        
        return subject_pos, object_pos
    except Exception as e:
        return "ERROR", "ERROR"

def count_constituent_size(subject_constituent, object_constituent, verb_constituent):
    """
    Count the number of tokens in each constituent
    """
    subject_size = len(subject_constituent.split()) if subject_constituent.strip() else 0
    object_size = len(object_constituent.split()) if object_constituent.strip() else 0
    verb_size = len(verb_constituent.split()) if verb_constituent.strip() else 0
    
    return subject_size, object_size, verb_size

def count_syllable_weight(subject_constituent, object_constituent, verb_constituent):
    """
    Count syllables (= count vowels) in each constituent
    """
    russian_vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
    
    def count_vowels(text):
        return len([char for char in text if char in russian_vowels])
    
    subject_syllables = count_vowels(subject_constituent)
    object_syllables = count_vowels(object_constituent)
    verb_syllables = count_vowels(verb_constituent)
    
    return subject_syllables, object_syllables, verb_syllables

def get_verb_aspect(verb_lemma, nlp_spacy):
    """
    Get aspect of the verb using SpaCy morphological analysis
    Returns 'Perf' (perfective), 'Imp' (imperfective), or 'UNKNOWN'
    """
    try:
        doc = nlp_spacy(verb_lemma)
        if len(doc) > 0 and doc[0].morph:
            morph = doc[0].morph
            if 'Aspect=Perf' in str(morph):
                return 'Perf'
            elif 'Aspect=Imp' in str(morph):
                return 'Imp'
        return 'UNKNOWN'
    except Exception as e:
        return 'ERROR'

def has_preceding_question(context):
    """
    Check if the last (closest to target) speech sentence in context ends with '?'
    Returns True if yes, False otherwise
    """
    if not context or not isinstance(context, str):
        return False
    
    speech_lines = []
    for line in context.split('\n'):
        if line.startswith('SPEECH:'):
            speech_content = line[7:].strip()  # Remove 'SPEECH:' prefix
            speech_lines.append(speech_content)
    
    # Check if the last speech line ends with '?'
    if speech_lines:
        last_speech = speech_lines[-1].strip()
        return last_speech.endswith('?')
    
    return False

def has_negation(sentence, verb_head, nlp_spacy):
    """
    Check if the verb is negated (preceded by 'не' or 'ни')
    Returns True if negated, False otherwise
    """
    try:
        doc = nlp_spacy(sentence)
        
        # Find the verb token
        verb_token = None
        for token in doc:
            if token.text.lower() == verb_head.lower():
                verb_token = token
                break
        
        if not verb_token:
            return False
        
        # Check if there's a negation particle that is close and dependent on the verb
        for token in doc:
            if token.lemma_.lower() in ['не', 'ни']:
                is_close = abs(token.i - verb_token.i) <= 2  # check if negation is close to the verb (within 2 tokens)

                is_dependent = (token.head == verb_token or  # check if negation is syntactically dependent on the verb
                               token.dep_ in ['advmod', 'neg'] and token.head == verb_token)
                
                if is_close and is_dependent:
                    return True
        
        return False
    except Exception as e:
        return False

def get_clause_type(sentence, verb_head, nlp_spacy):
    """
    Determine if the clause is matrix (root) or embedded
    Returns 'matrix' if verb is root, 'embedded' if not
    """
    try:
        doc = nlp_spacy(sentence)
        
        verb_token = None
        for token in doc:
            if token.text.lower() == verb_head.lower():
                verb_token = token
                break
        
        if not verb_token:
            return 'UNKNOWN'
        
        # Check if the verb is the root of the sentence
        if verb_token.dep_ == 'ROOT':
            return 'matrix'
        else:
            return 'embedded'
    except Exception as e:
        return 'ERROR'

def get_ner_tags(subject_head, object_head, nlp_stanza):
    """
    Get Named Entity Recognition tags for subject and object heads using Stanza
    Returns tuple of (subject_ner, object_ner)
    """
    try:
        # Process subject head
        subject_doc = nlp_stanza(subject_head)
        subject_ner = "not_applicable"
        for sent in subject_doc.sentences:
            if sent.ents:
                subject_ner = sent.ents[0].type
                break
        
        # Process object head
        object_doc = nlp_stanza(object_head)
        object_ner = "not_applicable"
        for sent in object_doc.sentences:
            if sent.ents:
                object_ner = sent.ents[0].type
                break
        
        return subject_ner, object_ner
    except Exception as e:
        return "ERROR", "ERROR"

def add_extra_features(csv_file_path, nlp_spacy, nlp_stanza):

    df = pd.read_csv(csv_file_path)    
    new_columns = {
        'subject_pos': [],
        'object_pos': [],
        'subject_constituent_size': [],
        'object_constituent_size': [],
        'verb_constituent_size': [],
        'subject_syllable_weight': [],
        'object_syllable_weight': [],
        'verb_syllable_weight': [],
        'verb_aspect': [],
        'preceding_question': [],
        'has_negation': [],
        'clause_type': [],
        'subject_ner': [],
        'object_ner': []
    }
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        # 1. POS tags
        subject_pos, object_pos = get_pos_tags(
            row['subject_lemma'], row['object_lemma'], nlp_spacy
        )
        new_columns['subject_pos'].append(subject_pos)
        new_columns['object_pos'].append(object_pos)
        
        # 2. Constituent sizes
        subject_size, object_size, verb_size = count_constituent_size(
            row['subject_constituent'], row['object_constituent'], row['verb_constituent']
        )
        new_columns['subject_constituent_size'].append(subject_size)
        new_columns['object_constituent_size'].append(object_size)
        new_columns['verb_constituent_size'].append(verb_size)
        
        # 3. Syllable weights
        subject_syllables, object_syllables, verb_syllables = count_syllable_weight(
            row['subject_constituent'], row['object_constituent'], row['verb_constituent']
        )
        new_columns['subject_syllable_weight'].append(subject_syllables)
        new_columns['object_syllable_weight'].append(object_syllables)
        new_columns['verb_syllable_weight'].append(verb_syllables)
        
        # 4. Verb aspect
        aspect = get_verb_aspect(row['verb_lemma'], nlp_spacy)
        new_columns['verb_aspect'].append(aspect)
        
        # 5. Preceding question
        prec_question = has_preceding_question(row['context'])
        new_columns['preceding_question'].append(prec_question)
        
        # 6. Negation
        negation = has_negation(row['sentence'], row['verb_head'], nlp_spacy)
        new_columns['has_negation'].append(negation)
        
        # 7. Clause type
        clause_type = get_clause_type(row['sentence'], row['verb_head'], nlp_spacy)
        new_columns['clause_type'].append(clause_type)
        
        # 8. NER tags
        subject_ner, object_ner = get_ner_tags(row['subject_head'], row['object_head'], nlp_stanza)
        new_columns['subject_ner'].append(subject_ner)
        new_columns['object_ner'].append(object_ner)
    
    for col_name, col_data in new_columns.items():   # add new columns to dataframe

        df[col_name] = col_data
    
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    print(f"Extra features added! Updated {csv_file_path}")
    
    # feature stats
    print("\nFeature Statistics:")
    print(f"POS tags - Subject: {df['subject_pos'].value_counts().head(3).to_dict()}")
    print(f"POS tags - Object: {df['object_pos'].value_counts().head(3).to_dict()}")
    print(f"Verb aspects: {df['verb_aspect'].value_counts().to_dict()}")
    print(f"Preceding questions: {df['preceding_question'].value_counts().to_dict()}")
    print(f"Negation: {df['has_negation'].value_counts().to_dict()}")
    print(f"Clause types: {df['clause_type'].value_counts().to_dict()}")
    print(f"Subject NER: {df['subject_ner'].value_counts().head(3).to_dict()}")
    print(f"Object NER: {df['object_ner'].value_counts().head(3).to_dict()}")
    
    return df

def main(nlp_spacy, nlp_stanza):
    df = add_extra_features(MAIN_DATA_FILE, nlp_spacy, nlp_stanza)
    return df
