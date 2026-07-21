# FAQ Chatbot using NLP

## Project Overview
This project implements an intelligent FAQ chatbot for a university using Natural Language Processing (NLP). It loads a CSV-based FAQ dataset, preprocesses questions, converts them to TF-IDF vectors, and matches user questions to the most relevant FAQ using cosine similarity.

## Features
- FAQ dataset stored in CSV for easy updates
- NLP preprocessing with lowercase normalization, punctuation removal, stopword removal, tokenization, and lemmatization
- TF-IDF text representation
- Cosine similarity for question matching
- Confidence score for each match
- Console chatbot interface
- Optional Streamlit web interface
- Logging and chat history export

## Technologies Used
- Python 3.11+
- pandas
- numpy
- scikit-learn
- nltk
- streamlit (optional)

## Installation
1. Create a virtual environment:
   `python -m venv .venv`
2. Activate it:
   - Windows: `.venv\Scripts\activate`
3. Install dependencies:
   `pip install -r requirements.txt`

## Running the Project
### Console chatbot
```bash
python app.py
```

### Run tests
```bash
pytest -q
```

## Example Inputs and Outputs
Input:
```text
How do I register for courses?
```

Output:
```text
Students can register for courses through the student portal during the registration period.

Matched FAQ: How do I register for courses?
```

## Folder Structure
```text
faq_chatbot/
├── data/
│   └── faq.csv
├── models/
├── utils/
│   ├── preprocessing.py
│   ├── similarity.py
│   └── chatbot.py
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
```

## How Preprocessing Works
The preprocessing pipeline converts text to lowercase, removes punctuation and numbers, removes stopwords, tokenizes text, and lemmatizes tokens to normalize the wording.

## How Cosine Similarity Works
Cosine similarity measures the angle between two vectors. In this project, the FAQ questions are converted into TF-IDF vectors, and the user's question is compared against them to find the closest match.

## Future Improvements
- Add typo tolerance and spell correction
- Add a Streamlit web interface
- Support voice input and output
- Add synonym handling and advanced sentence embeddings
