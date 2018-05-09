from statistics import mean

import matplotlib
import redis
import time

from sklearn.preprocessing import normalize

from redis_int.RedisUtil import recv_s
import numpy as np
#matplotlib.use('cairo')
matplotlib.use('TkAgg')



def to_array(s, idx):
    return np.array(list(np.array(s)[:, idx]), dtype=np.float32)


redis_local = redis.StrictRedis(host='192.168.0.25', port=6379, db=0)

cached_samples = recv_s(redis_local, key="samplecache", count=5000, poplimit=9999999999)

images = to_array(cached_samples, 2)
feature_images = to_array(cached_samples, 0)
total_weights = np.zeros(len(images))
for e in np.unique(feature_images):
    print(e)
    bit_feature_image = feature_images == e

    mean_images = np.mean(bit_feature_image, axis=0)
    diff_images = np.mean(np.mean(np.square(mean_images - bit_feature_image), axis=1), axis=1)
    weights = [np.square(diff_images)][0]
    total_weights += (weights - np.min(weights))/(np.max(weights) - np.min(weights))


import matplotlib.pyplot as plt
plt.imshow(images[np.argmax(total_weights)])
plt.colorbar()
plt.show()

time.sleep(10)
