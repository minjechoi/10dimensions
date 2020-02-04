# 10dimensions
Public repository containing the dataset and code for training the models in "Ten Social Dimensions of Conversations and Relationships" (WWW'20)

Steps

1. Run preprocess_data.py to transform the existing dataset in data/labeled-dataset.tsv to a BERT-friendly form

for using BERT

2. Run train_BERT.py to train the model for each of the 10 dimensions. The dimensions can be modified in the __main__ function.

3. Run the jupyter notebook "Test BERT classifier.ipynb" to load the trained model and apply it on any sentence.

for using LSTM

2. Download the GloVe embeddings at https://nlp.stanford.edu/projects/glove/ and put the unzipped file (.txt format) into weights/embeddings

3. Change the file directory name in the __main__ function at preprocess_embedding.py, then run the file. The result should be a .wv file generated

4. Run train_LSTM.py to train the LSTM models for each of the 10 dimensions.

5. Run the jupyter notebook "Test LSTM.ipynb" to load the trained model and apply it to any sentence.