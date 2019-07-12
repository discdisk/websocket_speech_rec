import chainer
import chainer.functions as F
from chainer.backends import cuda
from chainer.dataset import to_device
from chainer.dataset import concat_examples
from python_speech_features.base import delta
import numpy as np


class util():
    """docstring for utils"""

    def __init__(self, device, blank, out='word'):
        super(util, self).__init__()
        self.xp = cuda.cupy if device > -1 else np
        self.device = device
        self.blank = blank
        self.word_out = out == 'word'
        self.stacked_frames = 8
        self.skip_size = 3

    def converter(self, batch, device=-1):
        # alternative to chainer.dataset.concat_examples
        DATA_SHAPE = 40*3  # 40 log filterbank
        
        xs = [np.load(path).astype(np.float32) for path, _ in batch]
        delta_x = [delta(x,3) for x in xs]
        delta_delta_x = [delta(x,3) for x in delta_x]

        Xs = [to_device(self.device, np.concatenate((a,b,c),axis=1).astype(np.float32)) for a,(b,c) in zip(xs,zip(delta_x,delta_delta_x))]

        # Xs = [F.concat((X, self.xp.zeros(((self.stacked_frames + self.skip_size) - len(X), DATA_SHAPE), dtype=self.xp.float32)), axis=0) if len(X) < (self.stacked_frames + self.skip_size) else X for X in Xs]

        # Xs = [F.pad_sequence([X[i:i + self.stacked_frames] for i in range(0, len(X), self.skip_size)]).reshape(-1, DATA_SHAPE * self.stacked_frames) for X in Xs]

        word_label = [self.xp.asarray(lab[0]).astype(self.xp.int32) for _, lab in batch]
        char_lable = [self.xp.asarray(lab[1]).astype(self.xp.int32) for _, lab in batch]

        lable_batch = char_lable

        return Xs, lable_batch

    def ctc_loss(self, ys, lable_batch):
        lables = lable_batch
        (out_ys, input_length) = ys

        label_length = [len(l) for l in lables]
        label_length = self.xp.asarray(label_length, dtype=self.xp.int32)


        input_length = self.xp.asarray(input_length, dtype=self.xp.int32)

        word_lables = concat_examples(lables, self.device, padding=self.blank)
        word_loss = F.connectionist_temporal_classification(out_ys, word_lables, self.blank, input_length, label_length)

        # #confidence penalty
        # out_ys=F.stack(out_ys,axis=1)
        # out_ys=[F.softmax(out[:l]) for l,out in zip(input_length,out_ys)]

        # entropy=-sum([F.sum(out*F.log(out+1e-10)) for out in out_ys])/200

        # scale=0.2

        # word_loss=word_los-scale*entropy


        # if char_ys is not None:

        #     char_lables = concat_examples(char_lable, self.device, padding=self.blank)
        #     char_loss = F.connectionist_temporal_classification(char_ys, char_lables, self.blank, input_length, char_label_length)

        print(word_loss)

        return word_loss

    def cal_WER(self, x, y, predictor):
        result = predictor(x)
        result = F.stack(result, axis=1)
        out = []
        WER = 0
        for i, res in enumerate(result):
            out.append(res[:self.numframes[i]])
        for r, h in zip(y, out):
            WER += self._wer(r.data, decoder.decode(h.data)[0])

        return WER / len(y)

    def _wer(self, r, h):
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

        return d[len(r)][len(h)], len(h)


if __name__ == '__main__':
    u = util(0, 0)
    print(u._wer([1, 2, 2, 4, 2, 3, 4], [0, 4, 2, 3, 4]))
