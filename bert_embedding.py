import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer

# Load pretrained BERT model from TensorFlow Hub (use the right signature)
bert_model = hub.KerasLayer("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=False)

# Initialize a BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample sentence to embed
sentence = "I love programming with Python!"

# Tokenize the sentence
tokens = tokenizer(sentence, return_tensors='tf', max_length=128, truncation=True, padding='max_length')

# Extract the necessary inputs
input_word_ids = tokens['input_ids']
input_mask = tokens['attention_mask']

# BERT expects a dictionary with input_word_ids and input_mask
bert_inputs = {
    'input_word_ids': input_word_ids,
    'input_mask': input_mask
}

# Extract the output of the BERT model
bert_outputs = bert_model(bert_inputs)

# Get the [CLS] token embedding (represents the whole sentence)
sentence_embedding = bert_outputs['pooled_output']

print("Sentence Embedding Shape:", sentence_embedding.shape)
print("Sentence Embedding:", sentence_embedding)
