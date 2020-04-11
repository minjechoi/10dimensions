# 10dimensions
Public repository containing the dataset and code for training the models in "Ten Social Dimensions of Conversations and Relationships" (WWW'20) [paper](https://arxiv.org/abs/2001.09954)

### Requirements for code
- The requirements can be found [here](requirements.txt)
- Our LSTM models are based on pretrained GloVe word embeddings. Please refer to the instructions for setting the word embeddings [here](weights/embeddings/README.md)

### Preprocessing dataset

1. Run preprocess_data.py to transform the existing dataset in data/labeled-dataset.tsv to a BERT-friendly form

### Using pretrained models
- The instructions for downloading and using pretrained models can be found [here](weights/README.md)
- Once the pretrained weights are placed in the right directories, you can test the models on new sentences using the 'Test BERT' and 'Test LSTM' jupyter notebooks.

### Training your own models
- Run train_BERT.py or train_LSTM.py to train the model for each of the 10 dimensions. The dimensions can be modified in the __main__ function.

### Citation
If using the package for research purposes, please cite the following work. (Will be updated upon publication)

@misc{choi2020social,
    title={Ten Social Dimensions of Conversations and Relationships},
    author={Minje Choi and Luca Maria Aiello and Krisztian Zsolt Varga and Daniele Quercia},
    year={2020},
    eprint={2001.09954},
    archivePrefix={arXiv},
    primaryClass={cs.CY}
}
