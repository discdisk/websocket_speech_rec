
import chainer.functions as F
import chainer.links as L
import chainer
import numpy as np
from chainer.backends import cuda


def make_mask(i,win_size,batch_size,pad_size,n_mid_units,mask_length):
    mask=cuda.cupy.zeros((win_size,batch_size,n_mid_units),dtype=np.float32)
    if i<pad_size:
        mask[:pad_size-i] -= cuda.cupy.inf
    else:
        for b,l in enumerate(mask_length):
            if l >  i > l-win_size:
                mask[-i+l-win_size:,b,:] -= cuda.cupy.inf
    return mask

class RNN(chainer.Chain):

    def __init__(self, n_lstm_layers, n_mid_units, n_out, win_size, batch_size, att_units_size, dropout=0.5):
        super(RNN, self).__init__()
        ### actual number of lstm layers is 2*n_lstm_layers   ###

        initializer = chainer.initializers.Normal()


        ######   local attention related   #####
        xp = cuda.cupy
        self.Zu_init = xp.zeros((batch_size, n_out), dtype=np.float32)
        self.pad_size = int((win_size - 1) / 2)
        self.pad_zero = xp.zeros((self.pad_size, n_mid_units), dtype=self.xp.float32)
        self.n_mid_units = n_mid_units
        self.win_size = win_size
        self.batch_size = batch_size

        ########################################


        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units, initialW=initializer)

            self.encoder1 = L.NStepLSTM(1, n_mid_units, n_mid_units, dropout)
            self.encoder2 = L.NStepLSTM(1, n_mid_units, n_mid_units, dropout)
            self.encoder3 = L.NStepLSTM(1, n_mid_units, n_mid_units, dropout)

            self.lstm2 = L.NStepLSTM(n_lstm_layers-3, n_mid_units, n_mid_units, dropout)

            self.output = L.Linear(n_mid_units, n_out, initialW=initializer)

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
        ys = [self.output(y) for y in ys]


        ys = F.pad_sequence(ys)
        result = list(F.stack(ys, axis=1))


        return result, input_length


class Attention(chainer.Chain):
    """docstring for Attention"""

    def __init__(self, n_mid_units, n_out, win_size, batch_size, att_units_size, device=-1):
        super(Attention, self).__init__()

        self.win_size = win_size
        self.batch_size = batch_size
        self.att_units_size = att_units_size
        self.n_mid_units = n_mid_units

        initializer = chainer.initializers.LeCunNormal()

        with self.init_scope():
            self.last_out = L.Linear(None, self.att_units_size, initialW=initializer)
            self.hidden_layer = L.Linear(n_mid_units, self.att_units_size, initialW=initializer)
            self.att_cal = L.Linear(self.att_units_size, n_mid_units, initialW=initializer)

    def __call__(self, gt, last, mask):

        hx = self.hidden_layer(gt, n_batch_axes=2)

        Z = self.last_out(last)
        Z = F.broadcast_to(Z, (self.win_size, self.batch_size, self.att_units_size))

        attend = hx + Z

        attend =self.att_cal(
                F.tanh(attend.reshape(-1, self.att_units_size))
            ).reshape(-1, self.batch_size, self.n_mid_units)+mask


        attend = F.softmax(attend,axis=0)


        context = attend * gt

        context = F.sum(context, axis=0, keepdims=False) * self.win_size

        return context


class NstepLSTM_normalization(chainer.ChainList):
    """docstring for NstepLSTM_normalization"""

    def __init__(self, n_layers, in_size, out_size, dropout, **kwargs):
        super(NstepLSTM_normalization, self).__init__()
        with self.init_scope():
            for i in range(n_layers):
                self.add_link( LSTM_normalization(in_size, out_size, dropout))

    def __call__(self, xs):
        for child in self.children():
            xs = child(xs)
        return xs

class LSTM_normalization(chainer.Chain):
    """docstring for LSTM_normalization"""
    def __init__(self, in_size, out_size, dropout):
        super(LSTM_normalization, self).__init__()
        initializer = chainer.initializers.Normal()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1,
                                        in_size,
                                        out_size,
                                        dropout)
            self.layer_norm = L.LayerNormalization(out_size,eps=1e-5, initial_gamma=initializer)

    def __call__(self,xs):
        _, _, ys = self.lstm(None, None, xs)
        return [self.layer_norm(y) for y in ys]
        # return ys
    


class time_reduce(L.Linear):
    def __init__(self, *args, **kwargs):
        super(time_reduce, self).__init__(*args, **kwargs)


    def __call__(self, xs):
        return [self._call(x) for x in xs]

    def _call(self,x):
        length = int(x.shape[0] / 2) * 2
        x = x[:length]
        x_2d = x.reshape((-1, 2 * x.shape[-1]))
        out_2d = super(time_reduce, self).__call__(x_2d)
        # out_3d = out_2d.reshape(x.shape[:-1] + (out_2d.shape[-1], ))
        # (B, S, W)
        return out_2d
