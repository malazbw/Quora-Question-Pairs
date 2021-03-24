# Quora-Question-Pairs

## Problem
- Classify whether question pairs are duplicates or not.

## Approach

- Siamese Manhattan LSTM (MaLSTM) with help of - Siamese Recurrent Architectures for Learning Sentence Similarity paper with some modifciations.

## LINKS :
 * Kaggle competition - https://www.kaggle.com/c/quora-question-pairs
 * Dataset - https://www.kaggle.com/quora/question-pairs-dataset
 * MaLSTM paper - https://people.csail.mit.edu/jonasmueller/info/MuellerThyagarajan_AAAI16.pdf
 * GoogleNews-vectors-negative300 - https://github.com/mmihaltz/word2vec-GoogleNews-vectors
 

## RESULTS :

loss: 0.0999 - accuracy: 0.8737 - val_loss: 0.1195 - val_accuracy: 0.8377
<img width="293" alt="model" src="https://user-images.githubusercontent.com/14330105/112383071-f1b0d300-8cec-11eb-8348-f6c42b534431.png">
