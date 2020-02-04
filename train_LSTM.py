import os
from os.path import join
from preprocess_data import preprocessText


def loadDatasetForLSTM(dim,ver='train',data_dir = 'data/'):
    """
    File that releases the train/test/dev data as lists for future LSTM usage
    :param dim: one of the 10 dimensions
    :param ver: train/test/dev
    :param data_dir: where the data is stored
    :return: X and y, 2 lists
    """

    from sklearn.utils import resample
    from nltk.tokenize import TweetTokenizer
    tokenize = TweetTokenizer().tokenize

    assert ver in {'train','dev','test'}, "Incorrect version: please enter either 'train', 'test' or 'dev'"
    filename = join(data_dir,dim,'%s.tsv'%ver)
    assert os.path.exists(filename),"Error: file %s doesn't exist"%filename
    data = {0: [], 1: []}
    with open(filename) as f:
        for line in f:
            line = line.strip().split('\t')
            text = preprocessText(line[-1].lower())
            text = tokenize(text)
            if int(line[1]) == 1:
                data[1].append(text)
            else:
                data[0].append(text)
    print(len(data[0]), len(data[1]))
    if ver=='train':
        if len(data[1]) < len(data[0]):
            data[1] = resample(data[1], replace=True, n_samples=len(data[0]))
    X = data[0] + data[1]
    y = [0] * len(data[0]) + [1] * len(data[1])
    return X,y


def word2vec4gensim(file_dir):
    """

    :param file_dir:
    :return:
    """

    from gensim.models import KeyedVectors

    # load the vectors on gensim
    assert file_dir.endswith('.bin')|file_dir.endswith('.vec'), "Input file should be either a .bin or .vec"
    model = KeyedVectors.load_word2vec_format(file_dir,binary=file_dir.endswith('.bin'))
    # save only the .wv part of the model, it's much faster
    new_file_dir = file_dir.replace('.bin','.wv')
    model.wv.save(new_file_dir)
    # delete the original .bin file
    os.remove(file_dir)
    print("Removed previous file ",file_dir)

    # try loading the new file
    model = KeyedVectors.load(new_file_dir, mmap='r')
    print("Loaded in gensim! %d word embeddings, %d dimensions"%(len(model.vocab),len(model['a'])))
    return

def glove4gensim(file_dir):
    """
    A function that modifies the pretrained GloVe file so it could be integrated with this framework
    [Note] You can download the vectors used in this code at
    https://nlp.stanford.edu/projects/glove/ (make sure to unzip the files)
    :param file_dir: file directory of the downloaded file
    e.g., file_dir='/home/USERNAME/embeddings/word2vec/GoogleNews-vectors-negative300.bin'
    :return: None
    """

    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    # load the vectors on gensim
    # assert file_dir.endswith('.txt'), "For downloaded GloVe, the input file should be a .txt"
    # glove2word2vec(file_dir,file_dir.replace('.txt','.vec'))
    # file_dir = file_dir.replace('.txt','.vec')
    model = KeyedVectors.load_word2vec_format(file_dir,binary=file_dir.endswith('.bin'))
    # save only the .wv part of the model, it's much faster
    new_file_dir = file_dir.replace('.vec','.wv')
    model.wv.save(new_file_dir)
    # delete the original .bin file
    os.remove(file_dir)
    print("Removed previous file ",file_dir)

    # try loading the new file
    model = KeyedVectors.load(new_file_dir, mmap='r')
    print("Loaded in gensim! %d word embeddings, %d dimensions"%(len(model.vocab),len(model['a'])))
    return

def train(dim):
    import torch
    from torch import nn, optim
    import numpy as np
    from collections import Counter
    from features import ExtractWordEmbeddings
    from preprocess_data import batchify,padBatch
    from models.lstm import LSTMClassifier
    from sklearn.utils import shuffle


    # hyperparameters
    embedding_dim = 300 # changes only with different word embeddings
    hidden_dim = 300
    max_epochs = 100
    is_cuda = True
    batch_size = 60
    lr = 0.00001
    n_decreases = 10
    save_dir = 'weights/LSTM/%s'%dim
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load datasets
    X_tr,y_tr = loadDatasetForLSTM(dim,'train')
    # X_t,y_t = loadDatasetForLSTM(dim,'test')
    X_d,y_d = loadDatasetForLSTM(dim,'dev')

    # load model and settings for training
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    if is_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_dict = model.state_dict()
    flag = True
    old_val = np.inf # previous validation error
    em = ExtractWordEmbeddings(emb_type='glove')
    loss_fn = nn.BCELoss()

    # train model
    epoch = 0
    cnt_decrease = 0
    while (flag):
        tr_loss = 0.0
        epoch += 1
        if (epoch > max_epochs) | (cnt_decrease > n_decreases):
            break
        # train
        model.train()
        X_tr, y_tr = shuffle(X_tr, y_tr)
        tr_batches = batchify(X_tr, y_tr, batch_size)
        for X_b, y_b in tr_batches:
            # cn = Counter(y_b)
            # if cn[1] > 0:
            #     ratio = cn[0] / cn[1]
            # else:
            #     ratio = len(y_b)
            # weights = torch.tensor([ratio if x == 1 else 1 for x in y_b]).float()
            # if is_cuda:
            #     weights = weights.cuda()
            inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_b])).float()
            targets = torch.tensor(y_b, dtype=torch.float32)
            if is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)  # error here
            optimizer.zero_grad()
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()

        print("[Epoch %d] train loss: %1.3f" % (epoch, tr_loss))

        # validate
        current_loss = 0.0
        X_d, y_d = shuffle(X_d, y_d)
        val_batches = batchify(X_d, y_d, batch_size)
        for X_b, y_b in val_batches:
            inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_b])).float()
            targets = torch.tensor(y_b, dtype=torch.float32)
            if is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)  # error here
            current_loss += loss.item()

        print("[Epoch %d] validation loss: %1.3f" % (epoch, current_loss))
        if current_loss<old_val:
            # if current round is better than the previous round
            best_state = model.state_dict() # save this model
            torch.save(best_state, join(save_dir,'best-weights.pth'))
            # print(model.state_dict()['lstm.weight_ih_l0'][0][:5])
            print("Updated model")
            old_val = current_loss
            cnt_decrease=0
        else:
            # if the current round is doing worse
            cnt_decrease+=1

        if cnt_decrease>=n_decreases:
            flag = False
    return

def test(dim):
    import torch
    from torch import nn, optim
    import numpy as np
    from features import ExtractWordEmbeddings
    from preprocess_data import batchify,padBatch
    from models.lstm import LSTMClassifier
    from sklearn.utils import shuffle
    from sklearn.metrics import roc_auc_score, recall_score, accuracy_score

    # hyperparameters
    is_cuda = True
    batch_size = 60
    embedding_dim = 300
    hidden_dim = 300
    weight_dir = 'weights/LSTM/%s'%dim
    weight_file = join(weight_dir,'best-weights.pth')
    assert os.path.exists(weight_file),"The file directory for the saved model doesn't exist"

    # load datasets
    X_t,y_t = loadDatasetForLSTM(dim,'test')

    # load model and settings for training
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)
    if is_cuda:
        model.cuda()

    em = ExtractWordEmbeddings(emb_type='glove')

    # validate
    y_scores = []
    X_t, y_t = shuffle(X_t, y_t)
    val_batches = batchify(X_t, y_t, batch_size)
    for X_b, y_b in val_batches:
        inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_b])).float()
        targets = torch.tensor(y_b, dtype=torch.float32)
        if is_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # print(inputs.size())
        outputs = model(inputs).tolist()
        # print(outputs[-1],len(outputs))
        y_scores.extend(outputs)
    y_preds = np.array(np.array(y_scores)>=0.5,dtype=int)
    auc = roc_auc_score(y_true=y_t, y_score=y_scores)
    rec = recall_score(y_true=y_t, y_pred=y_preds)
    acc = accuracy_score(y_true=y_t, y_pred=y_preds)
    print('AUC: ', round(auc, 2))
    print('REC: ', round(rec, 2))
    print('ACC: ', round(acc, 2))
    with open(join(weight_dir,'scores.txt'),'w') as f:
        f.write('AUC: %1.2f\n'%auc)
        f.write('REC: %1.2f\n' % rec)
        f.write('ACC: %1.2f\n' % acc)
    return


if __name__=='__main__':
    dims = [
            'social_support',
            'conflict',
            'trust',
            'fun',
            'similarity',
            'identity',
            'respect',
            'romance',
            'knowledge',
            'power'
            ]
    for dim in dims:
        train(dim=dim)
        test(dim=dim)
