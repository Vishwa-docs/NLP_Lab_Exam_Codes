---
---
# Steps
1. 30 mins : Create Pipeline
    + See this file
    + Search online (Kaggle and Github)
        + Muthu, Rahul, Ask loh and achu
    + See Gemini
    + Go through similar stuff to get an idea
    + [TextaCy](https://textacy.readthedocs.io/en/latest/quickstart.html)
    + [FuzzyWuzzy](https://www.analyticsvidhya.com/blog/2021/06/fuzzywuzzy-python-library-interesting-tool-for-nlp-and-text-analytics/)
    + Ask on LMS
1. 30 Mins : Get the dataset
    + Search on kaggle and github

# To Do
- lol, just overexplain stuff, and act like u know ur shit
- like the method was not simply use a model, it was a workaround ryt
- Fine tuning models what parameters (Ask around and see which parameters to tune)
    - Where did you apply what, what did you fine tune
    - Something like number of layers, input and output side, learning rate etc.
- Models
    - KeyBERT and LSTM
    - Encoder Decoders

## Theory

- Recollect how the gates and connections work (For RNN, LSTMs etc.)

---
---
# NLP Lab Exam
Model and the function will be given and input and output will be given. You need to develop a model and get an output for the input, with justification on how it works and hyperparameters
You will be given 2 applications and you need to integrate it. 
    Eg : Sentence Compression and Language Generation, key word generation and machine translation, sumarization and compression etc
Project will be evaluated along with the model lab.

---
---
# Study Plans
1. Try the applications in lab exercises
    + Try by yourself, Cover everything in the pdf file, see the others codes
    + Answers in 
        - Notion Page (Ex 7 and 8)
        - Others solutions Folder
    + Take snippets from that, learn the flow
    + Add the flow to readme
    + Other Links
        - [Full Preprocessing Pipeline](https://www.kaggle.com/code/balatmak/text-preprocessing-steps-and-universal-pipeline)
        - [Cleaning and Preprocessing](https://www.kaggle.com/code/colearninglounge/nlp-data-preprocessing-and-cleaning)
1. Tracking
    1. 
    3. 
1. See the online codes (Other approaches, how they did for Lab Applications)
    - 5 (Any easier way?)
    - 7 (Better Model + See notion page)
1. Understand the steps and the codes in general
    + Why we are using what?
1. List other NLP problems / Applications **(Ask Gemini to generate)** and solve
    + See Readme File
    + Check standard datasets
    + nlp applications without deep learning list (Ask chatgpt)
1. Try the sample Questions (Time yourself)
1. See achintya work for our application and learn transformers and models
1. See the sheet Rhea sent


---
---
# Steps (Pipeline for Approaching Problems)
> Keep asking when you should do what, and what to use

## Dataset Retrival and Pipline
- Check gpt and github first

## Text Cleaning
**Normalization**: This process involves converting all text to the same case (upper or lower), removing extra spaces, and so on.
This is the first step where we remove unnecessary and redundant data. 

+ This includes:
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

## Converting Words to Vectors (Word Embedding/Text Vectors)
1. One Hot Encoding
1. Bag of Words
1. TF-IDF
1. **Deep Learning Techniques** - Word2Vec, GloVe, FastText
+ N-Grams, CBOW **(SEE IF YOU MISSED ANYTHING ELSE)**

## Model Building
1. **Text to Sequence Conversion**: This is the process of converting text data into sequence of integers or into vector form.
1. **Padding**: Padding is performed after the text to sequence conversion. It is used to ensure that all sequences in a list have the same length.
1. **Model Building** :
    + RNN, LSTM, GRU
    + Transformer and Attention Models
    + Bidirectional LSTM
    + Encoder and decoder architecture
1. **Model Evaluation**: What metrics for which application?
    + [BLEU Score](https://www.youtube.com/watch?v=DejHQYAGb7Q)

## Advanced Models (Pretrained)
They take care of embeddings and stuff by themselves
BERT, RoBERTa, DistilBERT

---
---
---
# References
- [Git Repo 1](https://github.com/Mr-Appu)
- [Repo 2](https://github.com/Aniruth1011)
- [Muthu](https://github.com/MuthuPalaniappan925/NLP-Dump)
- [Achu](https://github.com/Achintya-Lakshmanan)
- [Loh](https://github.com/KLohithSaradhi)

# Applications of NLP (Use Cases)
- [ ] Spell Check
- [ ] Text Classification
    - [x] Sentiment Analysis - IMDB Movie Dataset
    - [ ] Spam Classification
    - [ ] Auto Tagger
- [ ] [Text Summarization](https://www.youtube.com/watch?v=XO97Uon83Os)
- [ ] Fake News Detection
- [ ] Offensive Language Identification
- [x] Keyword Extraction
- [ ] Sentence Compression
- [ ] CNN
- [ ] Machine Translation (NMT)
    - [ ] Variants of BERT to handle different languages
- [ ] [Topic Extraction](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html)
- [ ] Next word / sentence prediction
    - [ ] Parsing Tree (Given tokens and stuff, how to do)
- [ ] [Sarcasm Detection (DIFFICULT)](https://github.com/Suji04/NormalizedNerd/blob/master/Introduction%20to%20NLP/Sarcasm%20is%20very%20easy%20to%20detect%20GloVe%2BLSTM.ipynb)
- [ ] Topic Modelling
- [ ] Sequence Modelling

## Basic Applications
Are also used as steps in many applications
- [x] Named Entity Recognition (NER)
- [ ] POS Tagging
- [ ] Dependency Parsing (Markov Modelling)
- [ ] QnA System

## Advanced
1. Reccomendation Systems
    1. Similar Sentence Identifcation (Cosing Similarity and Euclidean distance)
1. Chatbot

---
---