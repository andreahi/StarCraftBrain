import pickle

import zlib
import time

import numpy as np

def send_zipped_pickle(socket, obj, key="trainingset", protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.rpush(key, z)

def send_s(socket, obj, key="trainingset", protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.sadd(key, z)

def recv_s(socket, key="trainingsample", count=1, poplimit=0):
    #while socket.llen(key) < count:
    #    time.sleep(0.1)
    if count == -1:
        count = socket.scard(key)
    data_l = socket.srandmember(key, number=count)

    while socket.scard(key) > poplimit:
            socket.spop(key)
    return [pickle.loads(zlib.decompress(x), encoding='latin1') for x in data_l ]

def recv_pop_s(socket, key="trainingsample", count=1):

    if count == -1:
        count = socket.scard(key)
    count = min(socket.scard(key), count)

    data_l = []

    while len(data_l) < count:
        data_l.append(socket.spop(key))
    return [pickle.loads(zlib.decompress(x), encoding='latin1') for x in data_l ]

def to_array(s):
    return np.array(list(np.array(s)), dtype=np.float16)

def recv_range(socket, key="trainingsample", count=1):
    while socket.llen(key) < count:
        time.sleep(0.1)
    data_l = socket.lrange(key, 0, count)
    socket.ltrim(key, count, -1)
    return [pickle.loads(zlib.decompress(x), encoding='latin1') for x in data_l ]

def recv_zipped_pickle(socket, key="trainingset", blocking=True, timeout=0):
    """inverse of send_zipped_pickle"""
    if blocking:
        data = socket.blpop(key, timeout=timeout)
        if data == None:
            raise Exception("Redis timeout")
        key, z = data
    else:
        key, z = socket.lpop(key, timeout=timeout)
    p = zlib.decompress(z)
    return pickle.loads(p, encoding='latin1')
