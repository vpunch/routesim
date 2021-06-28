import numpy as np
import torch as tch

from .common import *


class QNetwork(SaveableModel):
    def __init__(self,
                 nodes_num,
                 layers,
                 activ_name,
                 embs_cfg,
                 addit_inputs=[]):
        super().__init__()

        self.embs_name = embs_cfg['name']
        if self.embs_name == 'oh':
            embs_cfg['dim'] = nodes_num
            in_dim = embs_cfg['dim'] * 3
        else:
            in_dim = embs_cfg['dim'] * 2

        label_tail = (f'_{"-".join(map(str, layers))}'
                      f'_{activ_name}' 
                      f'_{embs_cfg["name"]}-{embs_cfg["dim"]}')

        for inp in addit_inputs:
            if inp['name'] == 'amatrix':
                inp['dim'] = nodes_num ** 2
                in_dim += inp['dim']
            else:
                raise Exception('Unknown additional input: ' + name)

            label_tail += f'_{inp["name"]}-{inp["dim"]}'

        self.addit_inputs = addit_inputs

        self.label = f'qnetwork_{in_dim}{label_tail}'

        self.ff_net = FFNetwork(in_dim, 1, layers, activ_name)

    def forward(self, cur_embs, dst_embs, nbr_embs, *others):
        #[emb1,
        # emb2,
        # emb3]

        if self.embs_name != 'oh':
            lay_inp = [dst_embs - cur_embs, nbr_embs - cur_embs]
        else:
            lay_inp = [cur_embs, dst_embs, nbr_embs]

        for inp, other in zip(self.addit_inputs, others):
            if inp['name'] == 'amatrix':
                other[other > 0] = 1
                lay_inp.append(other)

        lay_inp = tch.tensor(np.concatenate(lay_inp, axis=1), dtype=tch.float)
        return self.ff_net(lay_inp)
