
import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda
from Modified_module.Modified_Layers import Additive_Attention


class RNN(chainer.Chain):

    def __init__(self, n_lstm_layers, n_mid_units, n_out, win_size, batch_size, att_units_size, frame_level=True, dropout=0.5):
        super(RNN, self).__init__()
        ### actual number of lstm layers is 2*n_lstm_layers   ###

        initializer = chainer.initializers.Normal()

        self.batch_size = batch_size
        ######   local attention related   #####
        xp = cuda.cupy
        self.Zu_init = xp.zeros((batch_size, n_out), dtype=np.float32)
        self.pad_size = int((win_size - 1) / 2)

        self.pad_zero = xp.zeros((self.pad_size, n_mid_units), dtype=np.float32)
        self.pad_inf = xp.full((self.pad_size, 1), -1e20, dtype=np.float32)

        self.win_size = win_size
        self.att_size = 1 if frame_level else n_mid_units

        ########################################


        with self.init_scope():
            self.l1        = L.Linear(None, n_mid_units, initialW=initializer)
            self.encoder1   = L.NStepLSTM(              1, n_mid_units, n_mid_units, dropout)
            self.encoder2   = L.NStepLSTM(              1, n_mid_units, n_mid_units, dropout)
            self.encoder3   = L.NStepLSTM(              1, n_mid_units, n_mid_units, dropout)


            self.lstm2     = L.NStepLSTM(n_lstm_layers-3, n_mid_units, n_mid_units, dropout)
            self.attend    = Additive_Attention(n_mid_units, win_size, batch_size, att_units_size)
            self.attend_ln = L.LayerNormalization(n_mid_units, initial_gamma=initializer)

            self.output    = L.Linear(n_mid_units*2, n_out, initialW=initializer)


    def _pad_transpose(self, input_list, shape, pad_num):
        pad  = self.xp.full(shape, pad_num, dtype=np.float32)
        input_list = [F.concat((F.concat((pad, m), axis=0), pad), axis=0) for m in input_list]
        input_list = F.pad_sequence(input_list, padding=pad_num)
        transposed_list = F.stack(input_list, axis=1)
        return transposed_list



    def __call__(self, xs):
        # forward calculation

        h1 = [F.relu(self.l1(x)) for x in xs]

        _,_,h1 = self.encoder1(None,None,h1)

        h1 = [F.max_pooling_2d(x[None][None],ksize=(3,1))[0][0] for x in h1]

        _,_,h1 = self.encoder2(None,None,h1)

        h1 = [F.max_pooling_2d(x[None][None],ksize=(2,1))[0][0] for x in h1]

        _,_,h1 = self.encoder3(None,None,h1)

        # h1 = [F.max_pooling_2d(x[None][None],ksize=(2,1))[0][0] for x in h1]


        _,_,ys = self.lstm2(None,None,h1)

        input_length = [len(y) for y in ys]

        

        ########  peform local attention   ######

        ### make mask ###
        mask = [self.xp.zeros((len(y),1),dtype=np.float32) for y in ys]
        mask = self._pad_transpose(mask, (self.pad_size,self.att_size), -1e20)

        gts  = self._pad_transpose(  ys, (self.pad_size,ys[0].shape[1]),     0)
        residual_h1  = self._pad_transpose(  h1, (self.pad_size,h1[0].shape[1]),     0)


        result = []
        last = self.Zu_init

        for i in range(len(gts) - self.pad_size * 2):

            local_mask = mask[i:i + self.win_size].data
            gt         = gts [i:i + self.win_size]
            context    = self.attend(gt, last, local_mask)

            context    = F.concat((context,residual_h1[i+3]))

            result.append(self.output(context))
            last = result[-1]

        return result, input_length


    def test(self,xs):
        result, input_length = self.__call__(xs)
        result = F.stack(result, axis=1)
        return [r[:l] for r,l in zip(result,input_length)]
    

