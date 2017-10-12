import redis

from redis_int.RedisUtil import recv_zipped_pickle

socket = redis.StrictRedis(host='localhost', port=6379, db=0)

try:
    print recv_zipped_pickle(socket, key="werwerwerwer", blocking=True, timeout=5)

except Exception:
    print "got an exception, but thats ok because I cought it :)"