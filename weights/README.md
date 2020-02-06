This is the directory where all the pretrained weights for models and word embeddings are stored.

#### Setting pretrained weights for LSTM models
1. Download the pretrained weights in [link](http://minjechoi.com/data/www20/LSTM.zip).

2. Unzip the file to find 10 folders, each containing both the pretrained weights (best-weights.pth) and score on the test set (scores.txt) for each dimension.

3. Put the 10 folders in LSTM, so that the best weights for "conflict" is in weights/LSTM/conflict/best-weights.pth

### Setting pretrained weights for BERT
1. For each dimension to use, download the pretrained weights
  - [conflict](http://minjechoi.com/data/www20/BERT/conflict.zip)
  - [social_support](http://minjechoi.com/data/www20/BERT/social_support.zip)
  - [romance](http://minjechoi.com/data/www20/BERT/romance.zip)
  - [power](http://minjechoi.com/data/www20/BERT/power.zip)
  - [knowledge](http://minjechoi.com/data/www20/BERT/knowledge.zip)
  - [respect](http://minjechoi.com/data/www20/BERT/respect.zip)
  - [fun](http://minjechoi.com/data/www20/BERT/fun.zip)
  - [trust](http://minjechoi.com/data/www20/BERT/trust.zip)
  - [identity](http://minjechoi.com/data/www20/BERT/identity.zip)
  - [similarity](http://minjechoi.com/data/www20/BERT/similarity.zip)

2. Extract the zip file and place all the contents in weights/BERT/##dimension_name##/.
  - e.g., for conflict, the weights should be stored in weights/BERT/conflict/pytorch_model.bin
  
