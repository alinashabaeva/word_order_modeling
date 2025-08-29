"""
Argument Structure Analysis Module

Adds thematic roles and predicate information:
1. Thematic roles of subject and object (Agent, Patient, Experiencer, etc.) (using OpenAI)
2. Verb type (psych, action, etc.) (using OpenAI)
3. Predicate type (2-place / 3-place) (using Stanza)
"""

import pandas as pd
import json
import time
from pathlib import Path
from openai import OpenAI
from typing import Dict
from config import MAIN_DATA_FILE, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_predicate_type_from_stanza(sentence, nlp_ru):
    """
    Determine predicate type (2-place/3-place) by counting arguments using Stanza parsing
    """
    if not nlp_ru:
        return '2-place'  # Default for transitive verbs
        
    try:
        doc = nlp_ru(sentence)
        
        main_verb = None
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'VERB' and word.deprel == 'root':
                    main_verb = word
                    break
            if main_verb:
                break
        
        if not main_verb:
            return '2-place'  # Default for transitive verbs
        
        # Count core arguments (subject, object, indirect object)
        argument_count = 0
        has_subject = False
        has_object = False
        has_indirect = False
        
        for sent in doc.sentences:
            for word in sent.words:
                # Check if this word is a dependent of the main verb
                if word.head == main_verb.id:
                    # Core arguments
                    if word.deprel in ['nsubj', 'nsubj:pass']:  # Subject
                        argument_count += 1
                        has_subject = True
                    elif word.deprel in ['obj', 'dobj']:  # Direct object
                        argument_count += 1
                        has_object = True
                    elif word.deprel in ['iobj', 'obl']:  # Indirect object/prepositional
                        # Only count as separate argument if it's not just a prepositional phrase
                        if word.upos in ['NOUN', 'PROPN', 'PRON']:
                            argument_count += 1
                            has_indirect = True
                    # Check for clausal arguments (count as 1 argument)
                    elif word.deprel in ['ccomp', 'xcomp', 'advcl']:  # Clausal complements
                        argument_count += 1
        
        # For transitive verbs analysis:
        # - 2-place: subject + object (e.g., "I see the cat")
        # - 3-place: subject + object + indirect object (e.g., "I give the book to John")
        
        if has_subject and has_object:
            if has_indirect or argument_count >= 3:
                return '3-place'
            else:
                return '2-place'
        else:
            return '2-place'  # Default for transitive verbs
            
    except Exception as e:
        print(f"Error parsing sentence with Stanza: {e}")
        return '2-place'  # Default fallback

def get_predicate_type_from_spacy(sentence, nlp):
    """
    Determine predicate type (2-place/3-place) by counting arguments using spaCy parsing
    """
    doc = nlp(sentence)
    
    main_verb = None
    for token in doc:
        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            main_verb = token
            break
    
    if not main_verb:
        return '2-place'
    
    core_relations = ['nsubj', 'nsubj:pass', 'obj', 'dobj', 'iobj', 'ccomp', 'xcomp']
    
    argument_count = 0
    for child in main_verb.children:
        if child.dep_ in core_relations:
            argument_count += 1
    
    # Return the actual predicate type based on argument count
    return f'{argument_count}-place'

def build_verb_type_prompt(verb: str) -> str:
    """
    Build a prompt for OpenAI to analyze a single verb type
    """
    prompt = f"""Analyze the semantic type of this Russian verb: {verb}

CATEGORIES:
- "psychological": mental states, emotions, cognition (любить, бояться, знать)
- "action": physical actions, changes (делать, строить, ломать)
- "communication": speech acts (говорить, рассказывать, объяснять)
- "perception": sensory experiences (видеть, слышать, смотреть)
- "motion": movement verbs (идти, ехать, лететь)
- "causative": causing changes (заставлять, помогать, позволять)
- "other": doesn't fit above categories

Respond with ONLY the category name (e.g., "psychological")"""

    return prompt

def build_thematic_roles_prompt(sentence: str, verb_lemma: str, subject_lemma: str, object_lemma: str) -> str:
    """
    Build a prompt for OpenAI to analyze thematic roles for a single sentence
    """
    prompt = f"""Analyze thematic roles in this Russian sentence:

Sentence: {sentence}
Verb: {verb_lemma}
Subject: {subject_lemma}
Object: {object_lemma}

SUBJECT ROLES:
- "Agent": intentional doer of action
- "Experiencer": one who perceives/feels
- "Causer": causes something without intention
- "Theme": undergoes motion/change
- "Stimulus": triggers a reaction

OBJECT ROLES:
- "Patient": undergoes action and is affected
- "Theme": undergoes action but not necessarily changed
- "Stimulus": triggers experience
- "Recipient": receives something
- "Beneficiary": benefits from action
- "Experiencer": experiences something

Respond with ONLY the roles in format: subject_role|object_role
Example: Agent|Patient"""

    return prompt

def extract_verb_type(response_text: str) -> str:
    """
    Extract verb type from simple response
    """
    line = response_text.strip().lower()
    valid_types = ['psychological', 'action', 'communication', 'perception', 'motion', 'causative', 'other']
    
    # Find matching type
    for verb_type in valid_types:
        if verb_type in line:
            return verb_type
    
    return 'unknown'

def extract_thematic_roles(response_text: str) -> dict:
    """
    Extract thematic roles from single sentence response (subject_role|object_role)
    """
    line = response_text.strip().lower()
    
    valid_subject_roles = ['agent', 'experiencer', 'causer', 'theme', 'stimulus']
    valid_object_roles = ['patient', 'theme', 'stimulus', 'recipient', 'beneficiary', 'experiencer']
    
    if '|' in line:
        parts = line.split('|', 1)
        subject_part = parts[0].strip()
        object_part = parts[1].strip()
        
        subject_role = 'unknown'
        for role in valid_subject_roles:
            if role in subject_part:
                subject_role = role.capitalize()
                break
        
        object_role = 'unknown'
        for role in valid_object_roles:
            if role in object_part:
                object_role = role.capitalize()
                break
        
        return {'subject_role': subject_role, 'object_role': object_role}
    else:
        return {'subject_role': 'unknown', 'object_role': 'unknown'}

def query_openai(prompt: str, max_retries: int = 3) -> str:
    """
    Query OpenAI API with retry logic
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a linguistic expert analyzing Russian sentences. Follow the exact format requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"OpenAI API failed after {max_retries} attempts: {str(e)}")
                return ""
            print(f"Attempt {attempt + 1} failed, retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)
    
    return ""

def add_argument_structure_columns(df, nlp_spacy, nlp_ru=None):
    """
    Add argument structure columns to the dataframe using batch processing
    """
    df['predicate_type'] = 'unknown'
    df['verb_type'] = 'unknown'
    df['subject_role'] = 'unknown'
    df['object_role'] = 'unknown'

    total_rows = len(df)
    
    # 1. Get predicate types using Stanza
    print("Analyzing predicate types with Stanza...")
    for index, row in df.iterrows():
        sentence = str(row['sentence'])
        predicate_type = get_predicate_type_from_stanza(sentence, nlp_ru)
        df.loc[index, 'predicate_type'] = predicate_type
    
    # 2. Get verb types for unique verbs only
    print("Analyzing verb types with OpenAI (unique verbs only)...")
    unique_verbs = df['verb_lemma'].unique()
    verb_type_cache = {}
    
    for i, verb in enumerate(unique_verbs):
        print(f"Processing verb {i+1}/{len(unique_verbs)}: {verb}")
        
        prompt = build_verb_type_prompt(verb)
        response = query_openai(prompt)
        
        if response:
            verb_type = extract_verb_type(response)
            verb_type_cache[verb] = verb_type
        else:
            verb_type_cache[verb] = 'unknown'
        
        time.sleep(0.5)
    
    for index, row in df.iterrows():
        verb_lemma = str(row['verb_lemma'])
        df.loc[index, 'verb_type'] = verb_type_cache.get(verb_lemma, 'unknown')
    
    # 3. Get thematic roles line-by-line 
    for index, row in df.iterrows():
        if (index + 1) % 50 == 0 or (index + 1) == total_rows:
            print(f"Processing role {index + 1}/{total_rows}")
        
        sentence = str(row['sentence'])
        verb_lemma = str(row['verb_lemma'])
        subject_lemma = str(row['subject_lemma'])
        object_lemma = str(row['object_lemma'])
        
        roles_prompt = build_thematic_roles_prompt(
            sentence, verb_lemma, subject_lemma, object_lemma
        )
        roles_response = query_openai(roles_prompt)
        
        if roles_response:
            roles = extract_thematic_roles(roles_response)
            df.loc[index, 'subject_role'] = roles['subject_role']
            df.loc[index, 'object_role'] = roles['object_role']
        
        time.sleep(0.5) 
    return df

def main(nlp_spacy, nlp_ru=None):
    df = pd.read_csv(MAIN_DATA_FILE)
    df = add_argument_structure_columns(df, nlp_spacy, nlp_ru)
    df.to_csv(MAIN_DATA_FILE, index=False, encoding='utf-8')
    
    # statistics
    print("\n Argument Structure Distribution:")
    print("Predicate types:")
    print(df['predicate_type'].value_counts())
    print("\nVerb types:")
    print(df['verb_type'].value_counts())
    print("\nSubject roles:")
    print(df['subject_role'].value_counts())
    print("\nObject roles:")
    print(df['object_role'].value_counts())
    
    return len(df)
