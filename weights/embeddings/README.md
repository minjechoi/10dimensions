# Setting word embeddings

Our LSTM model uses pretrained GloVe word embeddings to represent the input words. Here are the steps for downloading and setting them in our framework.

1. Download any of the GloVe embeddings from [here](https://nlp.stanford.edu/projects/glove/). 
We used the glove.840B.300d.zip settings from Common Crawl.

2. Unzip the zip file to obtain a .txt file storing all the embeddings for the words.

3. Store the .txt file in weights/embeddings.

4. Run preprocess_embedding.py from the root directory. Make sure that in line 57, the path of the embedding is set correctly.
Doing so will produce a .wv file and a .wv.vectors.npy file.
- Note that although we provide examples of GloVe only, both word2vec or FastText are supported as well. You can use the word2vec4gensim() function instead, which is also implemented in preprocess_embedding.py
- Our pretrained models are based on the glove.840B.300d settings, so using different embeddings would require re-training the models.

5. in the train_LSTM.py or Test LSTM.ipynb files, words will be transformed into word embeddings using the features.ExtractWordEmbeddings class object.

