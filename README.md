# Word Order Analysis

This project analyzes word order patterns in Russian transitive sentences (corpus analysis), with a focus on understanding the factors underlying variation between canonical SVO and non-canonical structures.

## Features

### Core Analysis
- **Transitive Sentence Extraction**: Identifies transitive sentences from the dataset (movie scripts)
- **Word Order Classification**: Categorizes sentences as SVO, OVS, VSO, etc.
- **Context Extraction**: Extracts and analyzes the context occuring before the target sentence.
- **Linguistic Feature Extraction**: POS tags, constituent sizes, syllable weights, verb aspect etc
- **Animacy Analysis**: Determines animacy of subjects and objects
- **Argument Structure Analysis**: Analyzes thematic roles and argument structure

### Coreference Score Calculation
The project includes sophisticated coreference score computation that evaluates whether subjects, objects, and verbs are already familier to the speaker/listener (e.g., the same lemma occured in the preceding context)

- **Subject Coreference Score**: Measures how well the subject maintains reference in context
- **Object Coreference Score**: Measures how well the object maintains reference in context  
- **Verb Coreference Score**: Measures how well the verb maintains reference in context

#### Coreference Score Components
Each score is calculated using weighted features. We optimized the weights using grid search and selected the setting that achieved the highest accuracy in word order identification:

1. **Pronoun/Proper Noun Detection** (`w_pron_per = 1.2`)
   - Higher scores for pronouns and proper nouns
   
2. **Speech Context Matching** (`w_match_speech = 0.8`)
   - Exponential decay based on position in speech lines
   
3. **Comment Context Matching** (`w_match_comment = 0.6`)
   - Same lemma for comment lines
   
4. **Semantic Similarity in Speech** (`w_sim_speech = 0.5`)
   - FastText-based similarity with position decay
   
5. **Semantic Similarity in Comment** (`w_sim_comment = 0.2`)
   - FastText-based similarity for comment lines
   
6. **Abstractness Scale** (`w_abstract_scale = 0.1`)
   - If the word is abstract, it is evaluated
   
7. **Determiner Boost** (`determiner_boost = 1.0`)
   - Multiplicative boost for determiners



## File Structure

```
word_order_Russian/
├── src/
│   ├── main.py                 # Main pipeline
│   ├── coreference_score.py    # Coreference calculation 
│   ├── sent_extraction.py      # Sentence extraction
│   ├── WO_extraction.py        # Word order analysis
│   ├── context_extraction.py   # Context extraction
│   ├── extra_features.py       # Linguistic features
│   ├── animacy.py              # Animacy analysis
│   ├── argument_structure.py   # Argument structure
│   └── config.py               # Configuration (not included in repo)
├── data/
│   ├── raw/                    # Raw input files (not included in repo)
│   └── processed/              # Output files (not included in repo)
├── extra/                      # Additional resources
│   ├── cc.ru.300.bin          # FastText model (not included in repo)
│   └── Slovar.r.ya..s.indeksom.konkretnosti.slov.xlsx  # Abstractness scores (not included in repo)
├── requirements.txt            # Dependencies
└── README.md                  # This file
```
