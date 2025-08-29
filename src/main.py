import sys
from pathlib import Path
import stanza
import spacy

from sent_extraction import main as transit_sent_extraction
from WO_extraction import main as word_order_extraction
from context_extraction import main as context_extraction
from extra_features import main as extra_features_extraction
from animacy import main as animacy_extraction
from argument_structure import main as argument_structure_extraction
from coreference_score import main as coreference_extraction

if __name__ == "__main__":
    stanza.download('ru')
    nlp_stanza = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse,ner')
    nlp_spacy = spacy.load("ru_core_news_lg")
    
    print("\nStep 1: Extracting transitive sentences...")
    num_sentences, file_stats = transit_sent_extraction(nlp_stanza)
    print(f"\nTotal: {num_sentences} transitive sentences found in all files.")
    
    if num_sentences > 0:
        print("\nStep 2: Analyzing word order patterns...")
        word_order_extraction(nlp_stanza)
        
        print("\nStep 3: Extracting context for transitive sentences...")
        context_extraction()
        
        print("\nStep 4: Adding extra linguistic features...")
        extra_features_extraction(nlp_spacy, nlp_stanza)
        
        print("\nStep 5: Determining animacy for subjects and objects...")
        animacy_extraction(nlp_spacy)
        
        print("\nStep 6: Analyzing argument structure and thematic roles...")
        argument_structure_extraction(nlp_spacy, nlp_stanza)
        
        print("\nStep 7: Calculating coreference scores...")
        coreference_extraction(nlp_spacy)
        
        print("\nResults saved to: data/processed/sentence_extraction_results.csv")
    else:
        print("\nNo transitive sentences found. Analysis skipped.")
