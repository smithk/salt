import os
import numpy as np
import cPickle


def setup_console(margin=5):
    """Numpy console output setup."""
    try:
        console_height, console_width = os.popen('stty size', 'r').read().split()
        np.set_printoptions(linewidth=int(console_width) - margin)
    except:
        pass


def store_object(object_to_serialize, filename):
    try:
        cPickle.dump(object_to_serialize, open(filename, 'w'))
    except Exception as e:
        print("Problem saving file {0}: {1}".format(filename, e))


def load_serialized_object(path):
    try:
        serialized_object = cPickle.load(open(path))
        return serialized_object
    except Exception as e:
        message = "Error reading serialized object at {path}: {exception}"
        print(message.format(path, e))


def load_serialized_object_array(path, append_arrays=True):
    """Load serialized objects from a file."""
    from itertools import chain

    objects = []
    eof, other_exception = False, False
    input_file = open(path)
    while not (eof or other_exception):
        try:
            objects.append(cPickle.load(input_file))
        except EOFError:
            eof = True
        except:
            other_exception = True
    if append_arrays:
        objects = list(chain.from_iterable(objects))
    return objects
