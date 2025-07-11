# Name: valid
# Author: Reacubeth
# Time: 2021/8/25 10:30
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import argparse
import numpy as np
import torch
import pickle
import time
import datetime
import os
import random
import utils
# from cenet_model_p import CENET


def execute_valid(args, total_data, model,
                  data,
                  s_history, o_history,
                  s_label, o_label,
                  s_frequency, o_frequency):
    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []

    s_ranks2_gen = []
    o_ranks2_gen = []
    all_ranks2_gen = []

    s_ranks3_gen = []
    o_ranks3_gen = []
    all_ranks3_gen = []
    total_data = utils.to_device(torch.from_numpy(total_data))
    for batch_data in utils.make_batch(data,
                                       s_history,
                                       o_history,
                                       s_label,
                                       o_label,
                                       s_frequency,
                                       o_frequency,
                                       args.batch_size):
        batch_data[0] = utils.to_device(torch.from_numpy(batch_data[0]))
        batch_data[3] = utils.to_device(torch.from_numpy(batch_data[3])).float()
        batch_data[4] = utils.to_device(torch.from_numpy(batch_data[4])).float()
        batch_data[5] = utils.to_device(torch.from_numpy(batch_data[5])).float()
        batch_data[6] = utils.to_device(torch.from_numpy(batch_data[6])).float()

        with torch.no_grad():
            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, \
            _, _, _, \
            sub_rank2_gen, obj_rank2_gen, cur_loss2_gen, \
            sub_rank3_gen, obj_rank3_gen, cur_loss3_gen, \
            ce_all_acc = model(batch_data, 'Valid', total_data)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3


            s_ranks2_gen += sub_rank2_gen
            o_ranks2_gen += obj_rank2_gen
            tmp2_gen = sub_rank2_gen + obj_rank2_gen
            all_ranks2_gen += tmp2_gen

            s_ranks3_gen += sub_rank3_gen
            o_ranks3_gen += obj_rank3_gen
            tmp3_gen = sub_rank3_gen + obj_rank3_gen
            all_ranks3_gen += tmp3_gen

    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3, \
           s_ranks2_gen, o_ranks2_gen, all_ranks2_gen, s_ranks3_gen, o_ranks3_gen, all_ranks3_gen
