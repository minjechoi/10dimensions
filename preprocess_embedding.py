import os
from gensim.models import KeyedVectors

def word2vec4gensim(file_dir):
    """

    :param file_dir:
    :return:
    """

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

    from gensim.scripts.glove2word2vec import glove2word2vec

    # load the vectors on gensim
    assert file_dir.endswith('.txt'), "For downloaded GloVe, the input file should be a .txt"
    glove2word2vec(file_dir,file_dir.replace('.txt','.vec'))
    file_dir = file_dir.replace('.txt','.vec')
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

if __name__=='__main__':
    # the file should be stored in weights/embeddings/
    glove4gensim('weights/embeddings/glove.840B.300d.txt') # change file name if using different embeddings
    # for example, if you downloaded the glove.42B.300d.zip instead, change the input of this function to the .txt file extracted from there.
    
