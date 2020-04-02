import os
from os.path import join
from preprocess_data import preprocessText
import argparse

def main():
    print('pid: ',os.getpid())
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dims", default=None, help="Set to None (all dimensions by default), can be customized by selecting dimensions, with ',' signs")

    parser.add_argument("--do_train", action='store_true', help="Whether to do training")
    parser.add_argument("--do_eval", action='store_true', help="Whether to do validation")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for LSTM model")
    parser.add_argument('--hidden_dim', type=int, default=300, help="Size of hidden layer in LSTM model")
    parser.add_argument('--max_epochs', type=int, default=100, help="The maximum number of epochs to run (allows early stopping)")

    args = parser.parse_args()

    # specify which dimensions
    if args.dims:
        dims = args.dims.split(',')

    else:
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

    # run train/test
    for dim in dims:
        if args.do_train:
            train(dim, args)
        if args.do_eval:
            test(dim, args)

    return

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
    with open(filename,encoding='utf-8',errors='ignore') as f:
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


def train(dim, args):
    import torch
    from torch import nn, optim
    import numpy as np
    from features import ExtractWordEmbeddings
    from preprocess_data import batchify,padBatch
    from models.lstm import LSTMClassifier
    from sklearn.utils import shuffle


    # hyperparameters
    embedding_dim = 300 # changes only with different word embeddings
    hidden_dim = args.hidden_dim
    max_epochs = args.max_epochs
    is_cuda = True
    batch_size = 60
    lr = args.lr
    n_decreases = 10
    save_dir = 'weights/LSTM/%s'%dim
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """
    Loading train / validation datasets
    X_tr: a list of tokenized sentences
    y_tr: a list of 0 and 1
    """
    X_tr,y_tr = loadDatasetForLSTM(dim,'train') # a list of tokenized sentences
    X_d,y_d = loadDatasetForLSTM(dim,'dev')


    # load model and settings for training
    model = LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    if is_cuda:
        model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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
        # for each iteration, shuffles X_tr and y_tr and puts them into batches
        X_tr, y_tr = shuffle(X_tr, y_tr)
        tr_batches = batchify(X_tr, y_tr, batch_size)
        for X_b, y_b in tr_batches:
            # X_b is still a list of tokenized sentences (list of list of words)
            optimizer.zero_grad()
            """
            obtain_vectors_from_sentence(sent=list of words, include_unk=True)
            : changes each word into an embedding, and returns a list of embeddings
            padBatch(list of embedding lists, max_seq=None)
            : for each batch, returns a tensor fixed to the max size, applies zero padding
            """
            inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_b])).float()
            # here, inputs become a tensor of shape (B * seq_len * dim)
            targets = torch.tensor(y_b, dtype=torch.float32)
            if is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)  # error here
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()

        print("[Epoch %d] train loss: %1.3f" % (epoch, tr_loss))

        # validate
        model.eval()
        current_loss = 0.0
        X_d, y_d = shuffle(X_d, y_d)
        val_batches = batchify(X_d, y_d, batch_size)
        with torch.no_grad():
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
            print("Updated model")
            old_val = current_loss
            cnt_decrease=0
        else:
            # if the current round is doing worse
            cnt_decrease+=1

        if cnt_decrease>=n_decreases:
            flag = False
    return

def test(dim, args):
    import torch
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
    hidden_dim = args.hidden_dim
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
    model.eval()
    with torch.no_grad():
        for X_b, y_b in val_batches:
            inputs = torch.tensor(padBatch([em.obtain_vectors_from_sentence(sent, True) for sent in X_b])).float()
            targets = torch.tensor(y_b, dtype=torch.float32)
            if is_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs).tolist()
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
    main()