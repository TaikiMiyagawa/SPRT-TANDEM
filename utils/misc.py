import os, yaml
import numpy as np
import tensorflow as tf 

def load_yaml(yaml_path):
    assert os.path.exists(yaml_path), "Yaml path does not exist: " + yaml_path
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def set_gpu_devices(gpu):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_visible_devices(physical_devices[gpu], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[gpu], True)


def make_directories(path):
    if not os.path.exists(path):
        print("Path '{}' does not exist.".format(path))
        print("Make directory: " + path)
        os.makedirs(path)
        
    
def fix_random_seed(flag_seed, seed=None):
    if flag_seed:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        print("Numpy and TensorFlow's random seeds fixed: seed=" + str(seed))
    
    else:
        print("Random seed not fixed.")


def show_layers(model):
    """Shows layers in model.
    Args:
        model: A tf.keras.Model object.
    """
    print("================= Model contains the followding layers ================")
    for iter_layer in model.layers:
        print("Layer: ", iter_layer.name)
    print("=======================================================================")

