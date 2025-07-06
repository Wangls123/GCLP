# Name: util
# Author: Reacubeth
# Time: 2021/6/25 17:08
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import os
import numpy as np
import torch
import argparse
import jax.numpy as jnp
import typing_extensions


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def make_batch(a, b, c, d, e, f, g, batch_size, valid1=None, valid2=None):
    if valid1 is None and valid2 is None:
        for i in range(0, len(a), batch_size):
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size]]
    else:
        for i in range(0, len(a), batch_size):
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size],
                   valid1[i:i + batch_size], valid2[i:i + batch_size]]


def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()


def isListEmpty(inList):
    if isinstance(inList, list):
        return all(map(isListEmpty, inList))
    return False


def get_sorted_s_r_embed_limit(s_hist, s, r, ent_embeds, limit):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_len_non_zero = torch.where(s_len_non_zero > limit, to_device(torch.tensor(limit)), s_len_non_zero)

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist[-limit:]:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def get_sorted_s_r_embed(s_hist, s, r, ent_embeds):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    """
    s_idx: id of descending by length in original list.  1 * batch
    s_len_non_zero: number of events having history  any
    s_tem: sorted s by length  batch
    r_tem: sorted r by length  batch
    embeds: event->history->neighbor
    lens_s: event->history_neighbor length
    embeds_split split by history neighbor length
    s_hist_dt_sorted: history interval sorted by history length without non
    """
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")


def write2file(s_ranks, o_ranks, all_ranks, s_ranks_gen, o_ranks_gen, all_ranks_gen, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    s_ranks_gen = np.asarray(s_ranks_gen)
    s_mr_lk_gen = np.mean(s_ranks_gen)
    s_mrr_lk_gen = np.mean(1.0 / s_ranks_gen)

    # find max mr and mrr
    s_mr_lk = s_mr_lk if s_mr_lk > (s_mr_lk + s_mr_lk_gen) else (s_mr_lk + s_mr_lk_gen)
    s_mrr_lk = s_mrr_lk if s_mrr_lk > (s_mrr_lk + s_mrr_lk_gen) else (s_mrr_lk + s_mrr_lk_gen)

    print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    file_test.write("Subject test MRR (lk): {:.6f}".format(s_mrr_lk) + '\n')
    file_test.write("Subject test MR (lk): {:.6f}".format(s_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        avg_count_sub_lk_gen = np.mean((s_ranks_gen <= hit))
        print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk + avg_count_sub_lk_gen))
        file_test.write("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk + avg_count_sub_lk_gen) + '\n')

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    o_ranks_gen = np.asarray(o_ranks_gen)
    o_mr_lk_gen = np.mean(o_ranks_gen)
    o_mrr_lk_gen = np.mean(1.0 / o_ranks_gen)

    o_mr_lk = o_mr_lk if o_mr_lk > (o_mr_lk + o_mr_lk_gen) else (o_mr_lk + o_mr_lk_gen)
    o_mrr_lk = o_mrr_lk if o_mrr_lk > (o_mrr_lk + o_mrr_lk_gen) else (o_mrr_lk + o_mrr_lk_gen)

    print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Object test MR (lk): {:.6f}".format(o_mr_lk))
    file_test.write("Object test MRR (lk): {:.6f}".format(o_mrr_lk) + '\n')
    file_test.write("Object test MR (lk): {:.6f}".format(o_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        avg_count_obj_lk_gen = np.mean((o_ranks_gen <= hit))
        print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk + avg_count_obj_lk_gen))
        file_test.write("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk + avg_count_obj_lk_gen) + '\n')

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    all_ranks_gen = np.asarray(all_ranks_gen)
    all_mr_lk_gen = np.mean(all_ranks_gen)
    all_mrr_lk_gen = np.mean(1.0 / all_ranks_gen)

    all_mr_lk = all_mr_lk if all_mr_lk > (all_mr_lk + all_mr_lk_gen) else (all_mr_lk + all_mr_lk_gen)
    all_mrr_lk = all_mrr_lk if all_mrr_lk > (all_mrr_lk + all_mrr_lk_gen) else (all_mrr_lk + all_mrr_lk_gen)

    print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    file_test.write("ALL test MRR (lk): {:.6f}".format(all_mrr_lk) + '\n')
    file_test.write("ALL test MR (lk): {:.6f}".format(all_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        avg_count_all_lk_gen = np.mean((all_mr_lk_gen <= hit))
        print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk + avg_count_all_lk_gen))
        file_test.write("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk + avg_count_all_lk_gen) + '\n')
    return all_mrr_lk


class LearningRateCallable(typing_extensions.Protocol):

  def __call__(self, step: jnp.ndarray) -> jnp.ndarray:
    ...


def create_learning_rate_scheduler(
    factors: str = 'constant * linear_warmup * rsqrt_decay',
    base_learning_rate: float = 0.5,
    warmup_steps: int = 1000,
    decay_factor: float = 0.5,
    steps_per_decay: int = 20000,
    steps_per_cycle: int = 100000,
    step_offset: int = 0,
    min_learning_rate: float = 1e-8,
) -> LearningRateCallable:
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * linear_decay: linear decay from warmup_steps with decay_factor slope. Note
      this option implies 'constant * linear_warmup', and should not be used in
      in conjunction with `constant` or `linear_warmup` factors.
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.
    step_offset: int, an offset that the step parameters to this function are
      relative to.
    min_learning_rate: float, minimum learning rate to output. Useful for cases
      when a decay function is (mis)configured to decay to non-positive values.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step: jnp.ndarray) -> jnp.ndarray:
    """Step to learning rate function."""
    step = jnp.maximum(0, step - step_offset)
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'linear_decay':
        ret *= base_learning_rate * jnp.minimum(
            step / warmup_steps, 1.0 + decay_factor * (warmup_steps - step)
        )
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= decay_factor ** (step // steps_per_decay)
      elif name == 'cosine_decay':
        progress = jnp.maximum(
            0.0, (step - warmup_steps) / float(steps_per_cycle)
        )
        ret *= jnp.maximum(
            0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0)))
        )
      else:
        raise ValueError('Unknown factor %s.' % name)
    ret = jnp.maximum(ret, min_learning_rate)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn