import numpy as np
import os
import pickle
import gc
import tqdm
from scipy.sparse import csc_matrix

def load_quadruples(inPath, fileName, fileName2=None):
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
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_history_target(quadruples, s_history_event_o, o_history_event_s, actor, target=None):
    if target is not None:
        if target == 'label':
            if actor == 's':
                s_history_oid = []
                ss = quadruples[:, 0]
                rr = quadruples[:, 1]
                oo = quadruples[:, 2]
                for ix in tqdm.tqdm(range(quadruples.shape[0])):
                    s_history_oid.append([])
                    for con_events in s_history_event_o[ix]:
                        cur_events = con_events[:, 1].tolist()
                        s_history_oid[-1] += cur_events
                s_history_label_true = np.zeros((quadruples.shape[0], 1))

                for ix in tqdm.tqdm(range(quadruples.shape[0])):
                    if oo[ix] in s_history_oid[ix]:
                        s_history_label_true[ix] = 1
                return s_history_label_true
            else:
                o_history_sid = []
                ss = quadruples[:, 0]
                oo = quadruples[:, 2]
                for ix in tqdm.tqdm(range(quadruples.shape[0])):
                    o_history_sid.append([])
                    for con_events in o_history_event_s[ix]:
                        cur_events = con_events[:, 1].tolist()
                        o_history_sid[-1] += cur_events
                o_history_label_true = np.zeros((quadruples.shape[0], 1))
                for ix in tqdm.tqdm(range(quadruples.shape[0])):
                    if ss[ix] in o_history_sid[ix]:
                        o_history_label_true[ix] = 1
                return o_history_label_true
        else:
            if actor == 's':
                rr = quadruples[:, 1]
                s_history_related = np.zeros((quadruples.shape[0], num_e), dtype=float)
                for ix in tqdm.tqdm(range(quadruples.shape[0])):
                    for con_events in s_history_event_o[ix]:
                        idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
                        cur_events = con_events[idxx, 1].tolist()
                        s_history_related[ix][cur_events] += 1
                s_history_related = csc_matrix(s_history_related)
                return s_history_related
            else:
                rr = quadruples[:, 1]
                o_history_related = np.zeros((quadruples.shape[0], num_e), dtype=float)
                for ix in tqdm.tqdm(range(quadruples.shape[0])):
                    for con_events in o_history_event_s[ix]:
                        idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
                        cur_events = con_events[idxx, 1].tolist()
                        o_history_related[ix][cur_events] += 1
                o_history_related = csc_matrix(o_history_related)
                return o_history_related
    else:
        if actor == 's':
            s_history_oid = []
            ss = quadruples[:, 0]
            rr = quadruples[:, 1]
            oo = quadruples[:, 2]
            s_history_related = np.zeros((quadruples.shape[0], num_e), dtype=float)
            for ix in tqdm.tqdm(range(quadruples.shape[0])):
                s_history_oid.append([])
                for con_events in s_history_event_o[ix]:
                    idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
                    cur_events = con_events[idxx, 1].tolist()
                    s_history_oid[-1] += con_events[:, 1].tolist()
                    s_history_related[ix][cur_events] += 1
            s_history_label_true = np.zeros((quadruples.shape[0], 1))

            for ix in tqdm.tqdm(range(quadruples.shape[0])):
                if oo[ix] in s_history_oid[ix]:
                    s_history_label_true[ix] = 1
            s_history_related = csc_matrix(s_history_related)
            return s_history_label_true, s_history_related
        else:
            o_history_sid = []
            ss = quadruples[:, 0]
            rr = quadruples[:, 1]
            oo = quadruples[:, 2]
            o_history_related = np.zeros((quadruples.shape[0], num_e), dtype=float)
            for ix in tqdm.tqdm(range(quadruples.shape[0])):
                o_history_sid.append([])
                for con_events in o_history_event_s[ix]:
                    idxx = (con_events[:, 0] == rr[ix]).nonzero()[0]
                    cur_events = con_events[idxx, 1].tolist()
                    o_history_sid[-1] += con_events[:, 1].tolist()
                    o_history_related[ix][cur_events] += 1

            o_history_label_true = np.zeros((quadruples.shape[0], 1))
            for ix in tqdm.tqdm(range(quadruples.shape[0])):
                if ss[ix] in o_history_sid[ix]:
                    o_history_label_true[ix] = 1
            o_history_related = csc_matrix(o_history_related)
            return o_history_label_true, o_history_related

print('GDELT - Test Data Processing')

test_data, test_times = load_quadruples('', 'test.txt')
num_e, num_r = get_total_number('', 'stat.txt')

s_his = [[] for _ in range(num_e)]
o_his = [[] for _ in range(num_e)]
s_his_t = [[] for _ in range(num_e)]
o_his_t = [[] for _ in range(num_e)]
s_history_data_test = [[] for _ in range(len(test_data))]
o_history_data_test = [[] for _ in range(len(test_data))]
s_history_data_test_t = [[] for _ in range(len(test_data))]
o_history_data_test_t = [[] for _ in range(len(test_data))]
latest_t = 0
s_his_cache = [[] for _ in range(num_e)]
o_his_cache = [[] for _ in range(num_e)]
s_his_cache_t = [None for _ in range(num_e)]
o_his_cache_t = [None for _ in range(num_e)]

for i, test in enumerate(test_data):
    if i % 10000 == 0:
        print("test", i, len(test_data))
    t = test[3]
    if latest_t != t:
        for ee in range(num_e):
            if len(s_his_cache[ee]) != 0:
                s_his[ee].append(s_his_cache[ee].copy())
                s_his_t[ee].append(s_his_cache_t[ee])
                s_his_cache[ee] = []
                s_his_cache_t[ee] = None
            if len(o_his_cache[ee]) != 0:
                o_his[ee].append(o_his_cache[ee].copy())
                o_his_t[ee].append(o_his_cache_t[ee])
                o_his_cache[ee] = []
                o_his_cache_t[ee] = None
        latest_t = t
    s = test[0]
    r = test[1]
    o = test[2]
    s_history_data_test[i] = s_his[s].copy()
    o_history_data_test[i] = o_his[o].copy()
    s_history_data_test_t[i] = s_his_t[s].copy()
    o_history_data_test_t[i] = o_his_t[o].copy()
    if len(s_his_cache[s]) == 0:
        s_his_cache[s] = np.array([[r, o]])
    else:
        s_his_cache[s] = np.concatenate((s_his_cache[s], [[r, o]]), axis=0)
    s_his_cache_t[s] = t
    if len(o_his_cache[o]) == 0:
        o_his_cache[o] = np.array([[r, s]])
    else:
        o_his_cache[o] = np.concatenate((o_his_cache[o], [[r, s]]), axis=0)
    o_his_cache_t[o] = t

with open('test_history_sub.txt', 'wb') as fp:
    pickle.dump([s_history_data_test, s_history_data_test_t], fp)
with open('test_history_ob.txt', 'wb') as fp:
    pickle.dump([o_history_data_test, o_history_data_test_t], fp)

s_label_test, s_history_related_test = get_history_target(test_data, s_history_data_test, o_history_data_test, 's')
o_label_test, o_history_related_test = get_history_target(test_data, s_history_data_test, o_history_data_test, 'o')

with open('test_s_label.txt', 'wb') as fp:
    pickle.dump(s_label_test, fp)
with open('test_o_label.txt', 'wb') as fp:
    pickle.dump(o_label_test, fp)
with open('test_s_frequency.txt', 'wb') as fp:
    pickle.dump(s_history_related_test, fp)
with open('test_o_frequency.txt', 'wb') as fp:
    pickle.dump(o_history_related_test, fp)

gc.collect()

