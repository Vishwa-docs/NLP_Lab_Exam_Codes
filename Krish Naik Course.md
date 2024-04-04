NLP - Speech and Text processing

# Applications of NLP (Use Cases)
1. Spell Check
1. Next word / sentence prediction
1. Language Translation (NMT)
1. Text Summarization
1. Sarcasm Detection (DIFFICULT)
1. Spam Classification

## Advanced
1. Chatbot

---
---
# NLP Project Pipeline

1. **Data Preprocessing**: Techniques like tokenization, vectorization, etc.
    + Cleaning the data
    + Tokenization (Sentence to Words), Stemming and Lemmatization, Stopwords
1. **Feature Engineering**: Techniques like word embeddings, TF-IDF, etc.
    + Converting words to vectors
    + BoW, TF-IDF, n-grams
1. **Deep Techniques**:
    + Word2Vec, GloVe, FastText etc
1. **Exploratory Data Analysis (EDA)**: Understand patterns, trends, outliers, etc. This could involve creating visualizations, calculating statistics, etc.
1. **Model Building** : Can be ML or DL
    + RNN, LSTM, GRU
    + Transformers (BERT) and Attention Models
    + Bidirectional LSTM, Encoders
1. **Model Evaluation**: What metrics for which model?

---
---
# Steps
> Keep asking when you should do what, and what to use

## Text Cleaning
This is the first step where we remove unnecessary and redundant data. This includes:
    - Removing HTML tags
    - Removing punctuation
    - Removing numbers or digits
    - Removing special characters
    - Removing extra whitespaces
    - Removing stop words: Stop words are common words that do not contribute much to the content or meaning of a document (e.g., "the", "is", "in", "and").

## Tokenization
+ Converting Sentence into Words

## Stop Words Removal
+ Remove the unnecessary words that do not add any meaning to the sentence
+ usually repeated in a sentence
+ "not" can play an important role, so you can create your own list

## Stemming and Lemmatization
+ Text normalization techniques
+ **Stemming** - Removes the last few characters to get the root word. Fast but word *may not have any meaning*
+ **Lemmatization** - takes into consideration the morphological analysis of the words to get the root words *(Preserves the root)*


Until here is pre processing
---

3. **Normalization**: This process involves converting all text to the same case (upper or lower), removing extra spaces, and so on.

6. **Part of Speech (POS) Tagging**: This involves identifying the part of speech of every word in the text (nouns, verbs, adjectives, etc.).

7. **Named Entity Recognition (NER)**: This is the process of identifying and classifying named entities in the text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

8. **Word Embedding/Text Vectors**: Word embedding is the representation of text in the form of vectors. The underlying idea here is that similar words will have a minimum distance between their vectors.

9. **Text to Sequence Conversion**: This is the process of converting text data into sequence of integers or into vector form.

10. **Padding**: Padding is performed after the text to sequence conversion. It is used to ensure that all sequences in a list have the same length.
