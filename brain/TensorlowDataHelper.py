import tensorflow as tf
import numpy as np

from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

def py_func_decorator(output_types=None, output_shapes=None, stateful=True, name=None):
    def decorator(func):
        def call(*args):
            nonlocal output_shapes

            flat_output_types = nest.flatten(output_types)
            flat_values = tf.py_func(
                func,
                inp=args,
                Tout=flat_output_types,
                stateful=stateful, name=name
            )
            if output_shapes is not None:
                # I am not sure if this is nessesary
                output_shapes = nest.map_structure_up_to(
                    output_types, tensor_shape.as_shape, output_shapes)
                flattened_shapes = nest.flatten_up_to(output_types, output_shapes)
                for ret_t, shape in zip(flat_values, flattened_shapes):
                    ret_t.set_shape(shape)
            return nest.pack_sequence_as(output_types, flat_values)
        return call
    return decorator

def from_indexable(iterator, output_types, output_shapes=None, num_parallel_calls=1, stateful=True, name=None):
    ds = tf.data.Dataset.range(len(iterator))
    @py_func_decorator(output_types, output_shapes, stateful=stateful, name=name)
    def index_to_entry(index):
        return iterator[index]
    return ds.map(index_to_entry, num_parallel_calls=num_parallel_calls)