"""
Animacy Detection Module

Determines whether subject and object heads are animate or inanimate using:
1. spaCy morphological features
2. Named Entity Recognition (NER) tags
3. Predefined pronoun lists
"""

import pandas as pd
from pathlib import Path
from config import MAIN_DATA_FILE

ANIMATE_PRONOUNS = {'кто', 'никто', 'кое-кто', 'кто-то', 'кто-нибудь'}
INANIMATE_PRONOUNS = {'что', 'ничто', 'кое-что', 'что-то', 'что-нибудь'}

def get_animacy_from_spacy(word, nlp):
    """
    Determine animacy using spaCy morphological features
    Returns: 'animate', 'inanimate', or 'not_applicable'
    """
    doc = nlp(word)
    
    for token in doc:
        if token.morph.get("Animacy"):
            animacy = token.morph.get("Animacy")[0]
            if animacy == "Anim":
                return 'animate'
            elif animacy == "Inan":
                return 'inanimate'
    
    return 'not_applicable'

def get_animacy_from_ner(word, sentence_text, nlp):
    """
    Determine animacy using Named Entity Recognition
    Returns: 'animate' or 'not_applicable'
    """
    doc = nlp(sentence_text)
    
    word_lower = word.lower()
    
    for ent in doc.ents:
        if word_lower in ent.text.lower():
            if ent.label_ == "PER":  
                return 'animate'

    return 'not_applicable'

def get_animacy_from_pronouns(word):
    """
    Determine animacy from predefined pronoun lists
    Returns: 'animate', 'inanimate', or 'not_applicable'
    """
    word_lower = word.lower().strip()
    
    if word_lower in ANIMATE_PRONOUNS:
        return 'animate'
    elif word_lower in INANIMATE_PRONOUNS:
        return 'inanimate'
    
    return 'not_applicable'

def determine_animacy(word, sentence_text, nlp):
    """
    Animacy identification:
    1. Start with spaCy morphology as base
    2. Override with NER if applicable  
    3. Override with pronouns if applicable
    """
    # 1. spaCy morphological features (baseline)
    result = get_animacy_from_spacy(word, nlp)
    
    # 2. modify with NER if we find a person
    ner_animacy = get_animacy_from_ner(word, sentence_text, nlp)
    if ner_animacy != 'not_applicable':
        result = ner_animacy
    
    # 3. modify with pronouns if applicable
    pronoun_animacy = get_animacy_from_pronouns(word)
    if pronoun_animacy != 'not_applicable':
        result = pronoun_animacy
    
    return result

def add_animacy_columns(df, nlp):
    """
    Add subject_animacy and object_animacy columns to the dataframe
    """
    df['subj_animacy'] = 'not_applicable'
    df['obj_animacy'] = 'not_applicable'
        
    for index, row in df.iterrows():
        sentence_text = str(row['sentence'])
        subj_head = str(row['subject_head']).strip()
        obj_head = str(row['object_head']).strip()
        
        if subj_head and subj_head != 'nan':
            df.loc[index, 'subj_animacy'] = determine_animacy(subj_head, sentence_text, nlp)
        
        if obj_head and obj_head != 'nan':
            df.loc[index, 'obj_animacy'] = determine_animacy(obj_head, sentence_text, nlp)
    
    return df

def main(nlp_spacy):
    df = pd.read_csv(MAIN_DATA_FILE)
    df = add_animacy_columns(df, nlp_spacy)
    df.to_csv(MAIN_DATA_FILE, index=False, encoding='utf-8')
    
    #  statistics
    print("\n Animacy Distribution:")
    print("Subject animacy:")
    print(df['subj_animacy'].value_counts())
    print("\nObject animacy:")
    print(df['obj_animacy'].value_counts())
    
    return len(df)

