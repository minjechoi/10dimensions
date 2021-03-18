This is the directory where all the pretrained weights for models and word embeddings are stored.

#### Setting pretrained weights for LSTM models
1. Download the pretrained weights in [link](https://www.dropbox.com/s/6nshzky3qpfbn8w/LSTM.zip?dl=0).

2. Unzip the file to find 10 folders, each containing both the pretrained weights (best-weights.pth) and score on the test set (scores.txt) for each dimension.

3. Put the 10 folders in LSTM, so that the best weights for "conflict" is in weights/LSTM/conflict/best-weights.pth

### Setting pretrained weights for BERT
1. For each dimension to use, download the pretrained weights
  - [conflict](https://www.dropbox.com/s/bw7s8o3fj6zrlvt/conflict.zip?dl=0)
  - [social_support](https://www.dropbox.com/s/aocsxs3u1uvqcov/social_support.zip?dl=0)
  - [romance](https://www.dropbox.com/s/6byehtmvcffe2ae/romance.zip?dl=0)
  - [power](https://www.dropbox.com/s/olvvo8eit3kucv5/power.zip?dl=0)
  - [knowledge](https://www.dropbox.com/s/8vze340nn1ip86v/knowledge.zip?dl=0)
  - [respect](https://www.dropbox.com/s/bj0z7l7e425kamj/respect.zip?dl=0)
  - [fun](https://www.dropbox.com/s/c6hhdlx62juittc/fun.zip?dl=0)
  - [trust](https://www.dropbox.com/s/4em5mey4xbzwqcg/trust.zip?dl=0)
  - [identity](https://www.dropbox.com/s/50o4v5ara4zec49/identity.zip?dl=0)
  - [similarity](https://www.dropbox.com/s/mxz5vtgsztvwarv/similarity.zip?dl=0)

2. Extract the zip file and place all the contents in weights/BERT/##dimension_name##/.
  - e.g., for conflict, the weights should be stored in weights/BERT/conflict/pytorch_model.bin
  
