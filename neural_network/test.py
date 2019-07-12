import os
import argparse
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'
import chainer
from CTCwithAttention import load_model
import chainer.functions as F

from tools.utils import util
import numpy as np
import pickle


def load_data(batch_size):
    dataset=''
    word_dic = pickle.load(open('/home/chenjh/Desktop/csj/making_correct_data/APS_npData_word_dic.pkl', 'rb'))
    char_dic = pickle.load(open('/home/chenjh/Desktop/csj/making_correct_data/APS_npData_char_dic.pkl', 'rb'))

    # Data structure
    # list of dic
    #########################
    # {'ori_filename': file,
    # 'core_noncore': data_folder[:-1],
    # 'ori_sound':f'ori_sound_{save_count}{count}.npy',
    # 'fbank_feat':f'fbank_feat_{save_count}{count}.npy',
    # 'fbank_feat_mean_norm':f'fbank_feat_mean_norm_{save_count}{count}.npy',
    # 'frame_length':fbank_feat.shape[0],
    # 'PlainOrthographicTranscription':text,
    # 'PhoneticTranscription':PhoneticTranscription}



    test_data_file = np.load('/home/chenjh/Desktop/csj/making_correct_data/new_xml_logf40_meanNorm_APS_test.npy')
    path = '/home/chenjh/Desktop/csj/making_correct_data/new_xml_logf40_meanNorm_APS_test/fbank_feat_mean_norm/'
    testX = [path + test_data['fbank_feat_mean_norm'] for test_data in test_data_file]
    testY = [test_data['output_word'] for test_data in test_data_file]
    testZ = [test_data['output_char'] for test_data in test_data_file]

    test = chainer.datasets.TupleDataset(testX, chainer.datasets.TupleDataset(testY, testZ))
   
    test_iter = chainer.iterators.SerialIterator(test[:len(test)//batch_size*batch_size], batch_size, shuffle=False, repeat=False)
    print('test_iter loaded')
    return test_iter, word_dic, char_dic

def remove_blank(labels, blank=0):
    new_labels = []

    # combine duplicate
    previous = 0
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l

    # remove blank
    new_labels = [l for l in new_labels if l != blank]

    return np.array(new_labels)


def _wer(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)] / len(h)



def main():



    b_size = 1
    unit_size =320
    n_layers =5

    gpu = 0
    import time

    test_iter, word_dic, char_dic = load_data(batch_size=b_size)

    utils = util(gpu, word_dic['blank'])
    word_dic = {b:a for a,b in word_dic.items()}





    model = load_model(gpu, b_size,unit_size,n_layers, utils.ctc_loss, (len(word_dic), len(char_dic))).predictor

    wer = 0
    count = 0
    aaa=0
    bbb=0
    tttt=time.time()
    for data in test_iter:
        Xs, (word_label, char_lable) = utils.converter(data,gpu) 
        aaa+=sum([x.shape[0] for x in Xs])

        ys = model.test(Xs) 

        for yy, tt in zip(ys, word_label):#to_device(-1, word_label)
            bbb+=len(tt)
            out = remove_blank(F.argmax(yy, axis=1).data)
            out = [int(o) for o in out]
            temp = _wer(out, tt)
            count += 1
            wer += temp
            print(len(out),' '.join([word_dic[o] for o in out]))
            print(len(tt),' '.join([word_dic[o] for o in tt.tolist()]))
            print(temp)
            print('')

    print('total WER is : ',wer/count)
    print(aaa,bbb)
    print(tttt-time.time())
    

if __name__ == '__main__':
    chainer.config.train = False
    main()
    # os.system('shutdown')
