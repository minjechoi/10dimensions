# 10dimensions
Public repository containing the dataset and code for training the models in "Ten Social Dimensions of Conversations and Relationships" (WWW'20)

### Requirements for code
The requirements can be found [here](requirements.txt)

### Preprocessing dataset

1. Run preprocess_data.py to transform the existing dataset in data/labeled-dataset.tsv to a BERT-friendly form

### Using pretrained models
- The instructions for downloading and using pretrained models can be found at [link](weights/README.md)
- Once the pretrained weights are placed in the right directories, you can test the models on new sentences using the 'Test BERT' and 'Test LSTM' jupyter notebooks.

### Training your own models
- Run train_BERT.py or train_LSTM.py to train the model for each of the 10 dimensions. The dimensions can be modified in the __main__ function.