from salt.IO import readers

if __name__ == '__main__':
    args = "salt/code/data/standard_ml_sets/classification/wine.arff"
    a = readers.ArffReader(args)
