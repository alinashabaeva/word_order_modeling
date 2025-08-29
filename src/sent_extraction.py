"""
Transitive Sentences Extraction Module

Extract transitive sentences from the dataset using Stanza dependency parsing.

- Extract transitive sentnces (object in accusative case and subject in nominative case) only from the data annotated as 'speech';
- Remove questions (sentences ending with '?')
- Remove sentences in imperative mood;
- Remove sentences with verbs 'звать', 'интересовать', 'иметь';
- Remove sentence with relative pronouns as object or subject (e.g., 'то, что я скажу')

Output:
- a CSV file with the following columns:
    - sentence: the extracted transitive sentence
    - id: a unique identifier for the sentence (e.g., 'pulpfiction.tsv_13' where pulpfiction.tsv is the source file and 13 is the sentence number)
    - subject_head: the head of the subject (nominative case)
    - object_head: the head of the object (accusative case)
    - verb_head: the head of the verb
    - subject_lemma: the lemma of the subject
    - object_lemma: the lemma of the object
    - verb_lemma: the lemma of the verb
    - subject_constituent: the constituent of the subject
    - object_constituent: the constituent of the object
    - verb_constituent: the constituent of the verb
"""

import os
import pandas as pd
import re
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

EXCLUDED_VERBS = {'звать', 'интересовать', 'иметь'}
MODAL_VERBS = {'мочь', 'хотеть', 'должен', 'быть', 'стать', 'начать', 'продолжать'}
CLAUSE_BOUNDARY_RELATIONS = {'conj', 'ccomp', 'xcomp', 'advcl', 'acl', 'csubj', 'csubjpass'}
RELATIVE_PRONOUNS = {'что', 'который', 'кто'}

def find_tsv_files(data_folder):
    """
    Find all TSV files in the data folder and return their paths
    """    
    tsv_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.tsv'):
                filepath = os.path.join(root, file)
                tsv_files.append((filepath, file))
    return tsv_files

def process_single_tsv_file(filepath, filename):
    """
    Process a single TSV file and return its lines
    """
    file_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            file_lines.append((line.strip(), filename))
    return file_lines

def extract_sentences_with_column_d(all_lines):
    """
    Extract sentences that have non-empty column D data (=speech)
    """

    sentence_tokens = defaultdict(list)
    
    for line, source_file in all_lines:
        if not line or line.startswith('#'):  # Skip empty lines and header lines
            continue
        
        parts = line.split('\t') # Split by tabs
        
        # Skip lines that don't have enough columns
        if len(parts) < 4:
            continue

        token_id, _, word, column_d = parts[0], parts[1], parts[2], parts[3] # Column A, B, C, D(=speech)
        
        if '-' in token_id:
            try:
                sentence_num, token_num = token_id.split('-', 1)
                token_pos = int(token_num)  
                
                token_info = {
                    'token_id': token_id,
                    'word': word,
                    'column_d': column_d,
                    'token_pos': token_pos,
                    'source_file': source_file
                }
                
                sentence_key = f"{source_file}_{sentence_num}" # Add tokens to sentence
                sentence_tokens[sentence_key].append(token_info)
                    
            except ValueError:
                continue
    
    return sentence_tokens

def reconstruct_sentences(sentence_tokens):
    """
    Reconstruct sentences by combining words that HAVE data in column D (speech)
    """
    
    reconstructed_sentences = {}
    
    for sent_key, tokens in sentence_tokens.items():
        tokens.sort(key=lambda x: x['token_pos']) # Sort tokens by position
        
        # Only include words that have something in column D (not empty, not '_')
        words_with_d = []
        for token in tokens:
            if token['column_d'].strip() and token['column_d'] != '_' and token['word']:
                words_with_d.append(token['word'])
        
        if words_with_d:  
            sentence_text = ' '.join(words_with_d)
            
            sentence_text = re.sub(r'\s+([.,!?;:])', r'\1', sentence_text) # Remove space before punctuation
            
            reconstructed_sentences[sent_key] = {
                'text': sentence_text,
                'tokens': tokens,
                'source_file': tokens[0]['source_file']
            }
    
    return reconstructed_sentences

def get_constituent(word, sent):
    """Get all words that depend on the given word (constituent), 
    but filter out irrelevant (erroneous) dependents like auxiliaries, conjunctions, and punctuation"""
    constituent_words = []
    
    children_map = defaultdict(list)
    for w in sent.words:
        children_map[w.head].append(w)
    
    def should_include_in_constituent(dependent, head):
        """Determine if a dependent should be included in the constituent"""
        if dependent.upos == 'PUNCT': #  exclude punctuation
            return False
        
        if head.upos in ['NOUN', 'PRON']: # For noun/pronoun heads (objects/subjects), exclude auxiliaries, coordinating conjunctions, relative clauses
            if dependent.upos == 'AUX':
                return False
            if dependent.upos == 'CCONJ' and dependent.deprel == 'cc':
                return False
            if dependent.deprel in ['nsubj', 'obj'] and dependent.upos in ['NOUN', 'PRON']:
                return False
            if dependent.deprel in ['acl', 'acl:relcl']:
                return False
        
        return True
    
    def collect_dependents(word_id):
        for w in children_map.get(word_id, []):  # only actual children
            if should_include_in_constituent(w, word):
                constituent_words.append(w)
                collect_dependents(w.id) 

    constituent_words.append(word)
    collect_dependents(word.id)
    
    constituent_words.sort(key=lambda x: x.id)     # Sort by word order in sentence and join
    return ' '.join(w.text for w in constituent_words)

def get_verb_constituent(verb, sent):
    """
    Get verb constituent including auxiliaries or modals related to our specific verb (like 'буду делать', 'должен делать')
    """
    verb_words = [verb.text]
    
    word_map = {w.id: w for w in sent.words}
    
    current_head = verb.head
    current_word = verb
    visited_heads = set()  
    max_depth = 10 
    depth = 0
    
    while current_head != 0 and current_head not in visited_heads and depth < max_depth:
        visited_heads.add(current_head)
        head_word = word_map.get(current_head)
        
        if not head_word:
            break
            
        if current_word.deprel in CLAUSE_BOUNDARY_RELATIONS:
            break  # Don't cross clause boundaries, check within the same clause only
        
        is_aux = head_word.upos == 'AUX'
        is_modal_like = head_word.lemma in MODAL_VERBS and head_word.upos in {'VERB', 'ADJ'}

        if is_aux or is_modal_like:
            verb_words.append(head_word.text)
            current_word = head_word
            current_head = head_word.head
        else:
            break
        
        depth += 1
    
    # Also check for auxiliaries and modals that depend on this verb (going down the tree)
    for word in sent.words:
        if word.head == verb.id:
            is_aux_dep = word.upos == 'AUX'
            is_modal_dep = word.lemma in MODAL_VERBS and word.upos in {'VERB', 'ADJ'}
            is_aux_relation = word.deprel in {'aux', 'aux:pass', 'cop'}
            
            if (is_aux_dep or is_modal_dep) and (is_aux_relation or word.deprel == 'xcomp'):
                verb_words.append(word.text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_verb_words = []
    for word in verb_words:
        if word not in seen:
            unique_verb_words.append(word)
            seen.add(word)
    
    word_to_id = {w.text: w.id for w in sent.words}
    verb_words_sorted = sorted(unique_verb_words, key=lambda x: word_to_id.get(x, float('inf')))
    return ' '.join(verb_words_sorted)

def should_filter_sentence(sentence_text, analysis, verb):
    """
    Find sentences that should be removed:
    - with relative pronouns as object or subject (e.g., 'то, что я скажу');
    - with verbs 'звать', 'интересовать', 'иметь' (~quirky sunjects)
    - with questions (sentences ending with '?')
    - with sentences in imperative mood
    """
    
    if sentence_text.strip().endswith('?') or sentence_text.strip().endswith('?!'):
        return True
    
    if _is_relative_clause_construction(sentence_text, analysis): # Remove relative clause constructions like "то, что я скажу"
        return True
    if analysis['verb_lemma'] in EXCLUDED_VERBS: # Remove sentences containing specific verbs
        return True
    if verb.feats and 'Mood=Imp' in verb.feats: # Remove sentences with verbs in imperative mood
        return True
    
    return False

def _is_relative_clause_construction(sentence_text, analysis):
    """
    Detect if this is a relative clause construction like 'то, что я скажу'
    """
    if analysis['object_lemma'] in RELATIVE_PRONOUNS: # Check for comma-separated relative pronoun constructions
        obj_head = analysis['object_head']
        if f", {obj_head}" in sentence_text or f",{obj_head}" in sentence_text: # Look for pattern: word + comma + relative pronoun
            return True
    
    if analysis['subject_lemma'] in RELATIVE_PRONOUNS: 
        subj_head = analysis['subject_head']
        if f", {subj_head}" in sentence_text or f",{subj_head}" in sentence_text: # Look for pattern: word + comma + relative pronoun 
            return True
    
    return False

def analyze_transitivity(sentence_text, nlp):
    """
    Analyze sentence for transitivity using Stanza
    Returns list of all transitive verb combinations (subject in Nom AND object in Acc related to the same verb)
    """
    try:
        doc = nlp(sentence_text)
        
        all_analyses = []
        
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos == 'VERB': 
                    verb = word
                    subject = None
                    obj = None
                    
                    for dependent in sent.words:
                        if (dependent.head == verb.id and 
                            dependent.deprel == 'nsubj' and 
                            dependent.feats and 'Case=Nom' in dependent.feats):
                            subject = dependent
                        
                        elif (dependent.head == verb.id and 
                              dependent.deprel == 'obj' and 
                              dependent.feats and 'Case=Acc' in dependent.feats):
                            obj = dependent
                    
                    if subject and obj:
                        try:
                            subject_constituent = get_constituent(subject, sent)
                            object_constituent = get_constituent(obj, sent)
                            verb_constituent = get_verb_constituent(verb, sent)
                            
                            analysis = {
                                'subject_head': subject.text,
                                'object_head': obj.text,
                                'verb_head': verb.text,
                                'subject_lemma': subject.lemma,
                                'object_lemma': obj.lemma,
                                'verb_lemma': verb.lemma,
                                'subject_constituent': subject_constituent,
                                'object_constituent': object_constituent,
                                'verb_constituent': verb_constituent
                            }
                            
                            if not should_filter_sentence(sentence_text, analysis, verb):
                                all_analyses.append(analysis)
                        except Exception as constituent_error:
                            continue
        
        return all_analyses if all_analyses else None
        
    except Exception as e:
        print(f"Error processing sentence: {str(e)[:100]}...")
        return None

def process_file_for_transitivity(filepath, filename, nlp):
    """
    Process a single file for transitivity analysis
    """
    print(f"\nProcessing file: {filename}")
    
    file_lines = process_single_tsv_file(filepath, filename)
    print(f"  - Found {len(file_lines)} lines")
    
    # sentences with speech data
    sentence_tokens = extract_sentences_with_column_d(file_lines)
    print(f"  - Found {len(sentence_tokens)} sentence groups with speech data")
    
    reconstructed_sentences = reconstruct_sentences(sentence_tokens)
    print(f"  - Reconstructed {len(reconstructed_sentences)} sentences (=speech data)")
    
    # Only show warning if no sentences reconstructed
    if not reconstructed_sentences and sentence_tokens:
        print(f"  - WARNING: No sentences reconstructed! Check speech data format.")
    
    # Analyze transitivity
    file_results = []
    processed_count = 0
    error_count = 0
    
    for sent_key, sent_data in tqdm(reconstructed_sentences.items(), desc=f"  Analyzing {filename}"):
        sentence_text = sent_data['text']
        processed_count += 1
        
        try:
            analyses = analyze_transitivity(sentence_text, nlp)
            
            if analyses:
                for i, analysis in enumerate(analyses):
                    unique_id = f"{sent_key}_verb{i+1}" if len(analyses) > 1 else sent_key
                    result = {
                        'sentence': sentence_text,
                        'id': unique_id,
                        'source_file': filename,
                        'subject_head': analysis['subject_head'],
                        'object_head': analysis['object_head'],
                        'verb_head': analysis['verb_head'],
                        'subject_lemma': analysis['subject_lemma'],
                        'object_lemma': analysis['object_lemma'],
                        'verb_lemma': analysis['verb_lemma'],
                        'subject_constituent': analysis['subject_constituent'],
                        'object_constituent': analysis['object_constituent'],
                        'verb_constituent': analysis['verb_constituent']
                    }
                    
                    file_results.append(result)
            else:
                # No transitive analysis found
                pass
                    
        except Exception as e:
            error_count += 1
            continue
    
    print(f"  - Results: {len(file_results)} transitive sentences found, {error_count} errors")
    return file_results

def main(nlp):
    tsv_files = find_tsv_files(RAW_DATA_DIR)
    print(f"Found {len(tsv_files)} TSV files to process")
    
    all_results = []
    file_stats = {}
    
    # Process each file individually
    for filepath, filename in tsv_files:
        file_results = process_file_for_transitivity(filepath, filename, nlp)
        all_results.extend(file_results)
        file_stats[filename] = len(file_results)
    
    print(f"\nPer-file statistics:")
    for filename, count in file_stats.items():
        print(f"  - {filename}: {count} transitive sentences")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / 'sentence_extraction_results.csv'
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved to: {output_file}")
    else:
        print("No transitive sentences found to save.")
    
    return len(all_results), file_stats

if __name__ == "__main__":
    main()
