import os
import warnings
import pandas as pd
from pathlib import Path

# Try to import numpy and gensim, but make them optional
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    warnings.warn("NumPy not available, some features will be limited")

try:
    from gensim.models.fasttext import load_facebook_vectors
    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False
    warnings.warn("Gensim not available, FastText similarity features will be disabled")

try:
    from .config import MAIN_DATA_FILE, ABSTRACTNESS_FILE, EXTRA_DIR
except ImportError:
    from config import MAIN_DATA_FILE, ABSTRACTNESS_FILE, EXTRA_DIR

# Try to import stanza, but make it optional
try:
    import stanza
    _HAS_STANZA = True
except ImportError:
    _HAS_STANZA = False
    warnings.warn("Stanza not available, some features will be limited")

# Coreference scoring weights
best_w = {
    "w_pron_per":       1.2,
    "w_match_speech":   0.8,
    "w_match_comment":  0.6,
    "w_sim_speech":     0.5,
    "w_sim_comment":    0.2,
    "w_abstract_scale": 0.1,
    "determiner_boost": 1.0,  
}

def calculate_exponential_decay(position, n_tokens):
    """Calculate exponential decay based on position in sentence."""
    if n_tokens <= 0: 
        return 0.0
    if not _HAS_NUMPY:
        return 0.0
    return float(np.exp(-position / n_tokens))

def stanza_lemmas(line, nlp_ru):
    """Extract lemmas from text using Stanza."""
    if not isinstance(line, str) or not line.strip():
        return []
    if not _HAS_STANZA or nlp_ru is None:
        return [t.lower() for t in line.split() if t.strip()]
    try:
        doc = nlp_ru(line)
        toks = [token.words[0] for sent in doc.sentences for token in sent.tokens]
        return [t.lemma.lower() for t in toks if t.text.strip()]
    except Exception:
        return [t.lower() for t in line.split() if t.strip()]

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    if not _HAS_NUMPY:
        return 0.0
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-12
    return float(np.dot(vec1, vec2) / denom)

def check_pronoun_or_per(pos_tag, ner_tag):
    """Check if token is pronoun/proper noun or person entity."""
    pos = str(pos_tag).upper() if pd.notna(pos_tag) else ""
    ner = str(ner_tag).upper() if pd.notna(ner_tag) else ""
    return (pos in {"PRON","PROPN"} or ner == "PER")

def find_lemma_match(head_lemma, context, nlp_ru):
    """Find lemma match in speech or comment contexts. Returns (score, found_in_speech, found_in_comment)."""
    if not isinstance(context, str) or not isinstance(head_lemma, str) or not head_lemma:
        return 0.0, False, False
    
    head = head_lemma.lower()
    speech_score, comment_score = 0.0, 0.0
    found_in_speech, found_in_comment = False, False
    
    for line in context.split("\n"):
        lt = "SPEECH" if line.startswith("SPEECH:") else "COMMENT" if line.startswith("COMMENT:") else None
        if not lt: 
            continue
            
        lemmas = stanza_lemmas(line, nlp_ru)
        n = len(lemmas)
        if n == 0: 
            continue
            
        found_pos = None
        for pos in range(n-1, -1, -1):
            if lemmas[pos] == head:
                found_pos = pos
                break
                
        if found_pos is not None:
            decay = calculate_exponential_decay(found_pos, n)
            if lt == "SPEECH":
                speech_score = max(speech_score, decay)
                found_in_speech = True
            else:
                comment_score = max(comment_score, decay)
                found_in_comment = True
    
    # If found in both speech AND comment, combine them
    if found_in_speech and found_in_comment:
        combined_score = speech_score + comment_score
        return combined_score, True, True
    elif found_in_speech:
        return speech_score, True, False
    elif found_in_comment:
        return comment_score, False, True
    else:
        return 0.0, False, False

def find_similarity_match(head_lemma, context, ft_model, nlp_ru, threshold=0.5):
    """Find similarity match using FastText vectors. Returns (score, found_in_speech, found_in_comment)."""
    if not isinstance(context, str) or not isinstance(head_lemma, str) or not head_lemma:
        return 0.0, False, False
    if ft_model is None or not _HAS_GENSIM:
        return 0.0, False, False
        
    try:
        head_vec = ft_model[head_lemma.lower()]
    except KeyError:
        return 0.0, False, False
        
    speech_score, comment_score = 0.0, 0.0
    found_in_speech, found_in_comment = False, False
    
    for line in context.split("\n"):
        lt = "SPEECH" if line.startswith("SPEECH:") else "COMMENT" if line.startswith("COMMENT:") else None
        if not lt: 
            continue
            
        lemmas = stanza_lemmas(line, nlp_ru)
        n = len(lemmas)
        if n == 0: 
            continue
            
        max_sim, max_pos = 0.0, 0
        for pos, tok in enumerate(lemmas):
            try:
                sim = cosine_similarity(head_vec, ft_model[tok])
            except KeyError:
                continue
            if sim > max_sim:
                max_sim, max_pos = sim, pos
                
        if max_sim >= threshold:
            if lt == "SPEECH":
                decay = calculate_exponential_decay(max_pos, n)
                score = max_sim * decay
                if score > speech_score:
                    speech_score = score
                    found_in_speech = True
            else:
                if max_sim > comment_score:
                    comment_score = max_sim
                    found_in_comment = True
    
    # If found in both speech AND comment, combine them
    if found_in_speech and found_in_comment:
        combined_score = speech_score + comment_score
        return combined_score, True, True
    elif found_in_speech:
        return speech_score, True, False
    elif found_in_comment:
        return comment_score, False, True
    else:
        return 0.0, False, False

def calculate_abstractness_score(lemma, pos_tag, ner_tag, abstract_dict, scale=0.3, threshold=0.1429):
    """Calculate abstractness score - higher for more concrete nouns."""
    if not isinstance(lemma, str) or not lemma:
        return 0.0
        
    pos_ok = (str(pos_tag).upper() == "NOUN") if pd.notna(pos_tag) else False
    ner_ok = (str(ner_tag).upper() != "PER") if pd.notna(ner_tag) else True
    
    if not pos_ok or not ner_ok: 
        return 0.0
        
    v = abstract_dict.get(lemma.lower())
    if v is None: 
        return 0.0
        
    try:
        if _HAS_NUMPY and not np.isfinite(v):
            return 0.0
        # Basic check for non-numeric values
        if not isinstance(v, (int, float)):
            return 0.0
    except:
        return 0.0
        
    base = abs(v * scale)
    return float(max(0.0, threshold - base))

def check_determiner(constituent, head_token):
    """Check if head token is preceded by a determiner."""
    DETERMINERS = {
        'мой','моя','моё','мои','твой','твоя','твоё','твои',
        'его','её','наш','наша','наше','наши','ваш','ваша','ваше','ваши',
        'этот','эта','это','эти','тот','та','то','те','такой','такая','такое'
    }
    
    if not isinstance(constituent, str) or not isinstance(head_token, str):
        return False
        
    toks = constituent.strip().split()
    try:
        idx = toks.index(head_token)
    except ValueError:
        toks_low = [t.lower() for t in toks]
        try:
            idx = toks_low.index(head_token.lower())
        except ValueError:
            return False
            
    return idx > 0 and toks[idx-1].lower() in DETERMINERS

def calculate_coreference_score(prefix, row, nlp_ru, ft_model, abstract_dict):
    """Calculate coreference score using priority-based logic."""
    
    # Start with 0 score
    final_score = 0.0
    
    # 1. PRONOUN/PER method - if found, use 1.2 and stop
    if check_pronoun_or_per(row[f"{prefix}_pos"], row[f"{prefix}_ner"]):
        final_score = best_w["w_pron_per"]
    else:
        # 2. Lemma matching - if found in speech OR comment, use that score
        lemma_score, in_speech, in_comment = find_lemma_match(row[f"{prefix}_lemma"], row["context"], nlp_ru)
        if lemma_score > 0:
            if in_speech:
                final_score = lemma_score * best_w["w_match_speech"]
            else:  # in_comment
                final_score = lemma_score * best_w["w_match_comment"]
        
        # 3. FastText similarity - if found in speech OR comment, use that score
        sim_score, in_speech, in_comment = find_similarity_match(row[f"{prefix}_lemma"], row["context"], ft_model, nlp_ru)
        if sim_score > 0:
            if in_speech:
                sim_final = sim_score * best_w["w_sim_speech"]
            else:  # in_comment
                sim_final = sim_score * best_w["w_sim_comment"]
            
            # Compare with current score and keep the highest
            if sim_final > final_score:
                final_score = sim_final
        
        # 4. Abstractness - if found, compare with current score and keep the highest
        abs_score = calculate_abstractness_score(
            row[f"{prefix}_lemma"], row[f"{prefix}_pos"], row[f"{prefix}_ner"],
            abstract_dict, scale=0.3, threshold=0.1429
        )
        if abs_score > 0:
            abs_final = abs_score * best_w["w_abstract_scale"]
            if abs_final > final_score:
                final_score = abs_final
    
    # 5. Determiner boost - apply only to final score if criteria met
    if check_determiner(row[f"{prefix}_constituent"], row[f"{prefix}_head"]):
        final_score *= best_w["determiner_boost"]
    
    return float(final_score)

def calculate_verb_coreference_score(row, nlp_ru, ft_model):
    """Calculate coreference score for verb using priority-based logic."""
    
    # Start with 0 score
    final_score = 0.0
    
    # 1. Lemma matching - if found in speech OR comment, use that score
    lemma_score, in_speech, in_comment = find_lemma_match(row["verb_lemma"], row["context"], nlp_ru)
    if lemma_score > 0:
        if in_speech:
            final_score = lemma_score * best_w["w_match_speech"]
        else:  # in_comment
            final_score = lemma_score * best_w["w_match_comment"]
    
    # 2. FastText similarity - if found in speech OR comment, use that score
    sim_score, in_speech, in_comment = find_similarity_match(row["verb_lemma"], row["context"], ft_model, nlp_ru)
    if sim_score > 0:
        if in_speech:
            sim_final = sim_score * best_w["w_sim_speech"]
        else:  # in_comment
            sim_final = sim_score * best_w["w_sim_comment"]
        
        # Compare with current score and keep the highest
        if sim_final > final_score:
            final_score = sim_final
    
    return float(final_score)

def main(nlp_spacy=None):
    """Main function to calculate coreference scores."""
    print("Calculating coreference scores using priority-based logic...")
    
    # Load data
    if not MAIN_DATA_FILE.exists():
        print(f"Data file not found: {MAIN_DATA_FILE}")
        return
        
    df = pd.read_csv(MAIN_DATA_FILE)
    
    # Check required columns
    required_base_cols = [
        "subject_pos", "subject_ner", "object_pos", "object_ner",
        "subject_lemma", "object_lemma", "verb_lemma",
        "subject_constituent", "object_constituent",
        "subject_head", "object_head", "context", "word_order"
    ]
    
    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return
    
    # Stanza
    nlp_ru = None
    if _HAS_STANZA:
        try:
            nlp_ru = stanza.Pipeline('ru', processors='tokenize,pos,lemma', use_gpu=False, verbose=False)
            print("Stanza initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Stanza: {e}")
            print("Will use basic text processing instead")
    else:
        print("Stanza not available, using basic text processing")
    
    # FastText model
    ft_model = None
    if _HAS_GENSIM:
        try:
            from .config import FASTTEXT_MODEL
        except ImportError:
            from config import FASTTEXT_MODEL
        if FASTTEXT_MODEL.exists():
            try:
                ft_model = load_facebook_vectors(str(FASTTEXT_MODEL))
                print("FastText model loaded successfully")
            except Exception as e:
                print(f"Failed to load FastText model: {e}")
        else:
            print("FastText model not found, similarity features will be zeros")
    else:
        print("Gensim not available, FastText similarity features will be zeros")
    
    # Load abstractness dictionary
    abstract_dict = {}
    if ABSTRACTNESS_FILE.exists():
        try:
            df_dict = pd.read_excel(ABSTRACTNESS_FILE, sheet_name="nouns")
            abstract_dict = {
                str(r['word']).lower(): float(r['score'])
                for _, r in df_dict.iterrows()
                if pd.notna(r.get('word')) and pd.notna(r.get('score'))
            }
            print("Abstractness dictionary loaded successfully")
        except Exception as e:
            print(f"Failed to load abstractness dictionary: {e}")
    else:
        print("Abstractness file not found, abstractness features will be zeros")
    
    # Calculate coreference scores 
    print("Calculating subject coreference scores...")
    df["coref_score_subject"] = df.apply(
        lambda r: calculate_coreference_score("subject", r, nlp_ru, ft_model, abstract_dict), axis=1
    )
    
    print("Calculating object coreference scores...")
    df["coref_score_object"] = df.apply(
        lambda r: calculate_coreference_score("object", r, nlp_ru, ft_model, abstract_dict), axis=1
    )
    
    print("Calculating verb coreference scores...")
    df["coref_score_verb"] = df.apply(
        lambda r: calculate_verb_coreference_score(r, nlp_ru, ft_model), axis=1
    )
    
    df.to_csv(MAIN_DATA_FILE, index=False)
    print(f"Coreference scores calculated and saved to: {MAIN_DATA_FILE}")
    
    # statistics
    print("\nCoreference Score Summary:")
    try:
        if _HAS_NUMPY:
            print(f"Subject scores - Mean: {df['coref_score_subject'].mean():.3f}, Std: {df['coref_score_subject'].std():.3f}")
            print(f"Object scores  - Mean: {df['coref_score_object'].mean():.3f}, Std: {df['coref_score_object'].std():.3f}")
            print(f"Verb scores    - Mean: {df['coref_score_verb'].mean():.3f}, Std: {df['coref_score_verb'].std():.3f}")
        else:
            print(f"Subject scores - Mean: {df['coref_score_subject'].mean()}")
            print(f"Object scores  - Mean: {df['coref_score_object'].mean()}")
            print(f"Verb scores    - Mean: {df['coref_score_verb'].mean()}")
    except Exception as e:
        print(f"Could not calculate summary statistics: {e}")
        print("Raw scores calculated successfully")
    
    return df

if __name__ == "__main__":
    main()