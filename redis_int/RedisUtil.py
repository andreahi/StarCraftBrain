import pickle

import zlib


def send_zipped_pickle(socket, obj, key="trainingset", protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.rpush(key, z)

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
    return pickle.loads(p)