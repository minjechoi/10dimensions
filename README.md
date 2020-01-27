# 10dimensions
Public repository containing the dataset and code for training the models in "Ten Social Dimensions of Conversations and Relationships" (WWW'20)

Steps

1. Run preprocess_data.py to transform the existing dataset in data/labeled-dataset.tsv to a BERT-friendly form
2. Run train_BERT.py to train the model for each of the 10 dimensions. The dimensions can be modified in the __main__ function.
3. Run the jupyter notebook "Test BERT classifier.ipynb" to load the trained model and apply it on any sentence.