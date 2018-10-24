import json
import pickle
import pyarrow

from bson import Binary
from pymongo import MongoClient
import numpy as np

client = MongoClient("10.0.0.112")
db = client.trainingDB

def insert_split(s_0, s_1, s_2, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7,
                        a_8, r, v,
                        next_s, next_state0, next_state1, next_state2, db_name = "samplecache"):

    db[db_name].insert({"s_0":json.dumps(s_0), "s_1":json.dumps(s_1), "s_2":json.dumps(s_2), "a_0":json.dumps(a_0), "a_1":json.dumps(a_1), "a_2":json.dumps(a_2), "a_3":json.dumps(a_3), "a_4":json.dumps(a_4), "a_5":json.dumps(a_5), "a_6":json.dumps(a_6), "a_7":json.dumps(a_7),
                        "a_8": json.dumps(a_8), "r":json.dumps(r), "v":json.dumps(v),
                        "next_s": json.dumps(next_s), "next_state0": json.dumps(next_state0), "next_state1": json.dumps(next_state1), "next_state2": json.dumps(next_state2)})




def insert(data, db_name = "samplecache"):
    serialized_x = pyarrow.serialize(data).to_buffer()

    db[db_name].insert({"data":Binary(serialized_x.to_pybytes())})


def retrieve(samples = 1000, db_name = "samplecache"):
    data = []
    print("aggregating data")
    for _ in range(3):
        aggregate = db[db_name].aggregate([
            {"$sample": {"size": samples}}

        ], allowDiskUse=True)

        print("pickeling data")
        for e in list(aggregate):
            serialized_x = pyarrow.deserialize(e["data"])

            data.append(serialized_x)
        print("pickeling done")

    print("returning data")
    return data