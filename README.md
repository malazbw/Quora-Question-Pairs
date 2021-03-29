# Quora-Question-Pairs

## Problem
- Classify whether question pairs are duplicates or not.

## Approach

- Siamese Manhattan LSTM (MaLSTM) with help of - Siamese Recurrent Architectures for Learning Sentence Similarity paper with some modifciations.

## Modeling
<img width="352" alt="model" src="https://user-images.githubusercontent.com/14330105/112391461-3c841800-8cf8-11eb-9a0b-c278c824a7fc.png">


## LINKS :
 * Kaggle competition - https://www.kaggle.com/c/quora-question-pairs
 * Dataset - https://www.kaggle.com/quora/question-pairs-dataset
 * MaLSTM paper - https://people.csail.mit.edu/jonasmueller/info/MuellerThyagarajan_AAAI16.pdf
 * GoogleNews-vectors-negative300 - https://github.com/mmihaltz/word2vec-GoogleNews-vectors
 

## RESULTS :

loss: 0.0999 - accuracy: 0.8570  - val_loss: 0.1195 - val_accuracy: 0.8307



