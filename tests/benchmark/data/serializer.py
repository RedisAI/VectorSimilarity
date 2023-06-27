import numpy as np
from VecSim import *
import h5py
import os

# location = os.path.join(os.path.abspath('.'), 'tests', 'benchmark', 'data', '')
location = os.path.join(os.path.abspath('.'), '')
print('at: ', location)

DEFAULT_FILES = [
    {
        'filename': 'dbpedia-768',
        'nickname': 'dbpedia',
        'dim': 768,
        'metric': VecSimMetric_Cosine,
    },
    {
        'filename': 'dbpedia-768',
        'nickname': 'dbpedia',
        'dim': 768,
        'type': VecSimType_FLOAT64,
        'metric': VecSimMetric_Cosine,
    },
    {
        'filename': 'fashion_images_multi_value',
        'metric': VecSimMetric_Cosine,
        'multi': True,
    },
    {
        'filename': 'fashion_images_multi_value',
        'metric': VecSimMetric_Cosine,
        'type': VecSimType_FLOAT64,
        'multi': True,
    },
    {
        'filename': 'glove-25-angular',
        'nickname': 'glove',
        'M': 16,
        'efConstruction': 100,
        'metric': VecSimMetric_Cosine,
        'skipRaw': True,
    },
    {
        'filename': 'glove-50-angular',
        'nickname': 'glove',
        'M': 24,
        'efConstruction': 150,
        'metric': VecSimMetric_Cosine,
        'skipRaw': True,
    },
    {
        'filename': 'glove-100-angular',
        'nickname': 'glove',
        'M': 36,
        'efConstruction': 250,
        'metric': VecSimMetric_Cosine,
        'skipRaw': True,
    },
    {
        'filename': 'glove-200-angular',
        'nickname': 'glove',
        'M': 48,
        'efConstruction': 350,
        'metric': VecSimMetric_Cosine,
        'skipRaw': True,
    },
    {
        'filename': 'mnist-784-euclidean',
        'nickname': 'mnist',
        'M': 32,
        'efConstruction': 150,
        'metric': VecSimMetric_L2,
        'skipRaw': True,
    },
    {
        'filename': 'sift-128-euclidean',
        'nickname': 'sift',
        'M': 32,
        'efConstruction': 200,
        'metric': VecSimMetric_L2,
        'skipRaw': True,
    },
]

def serialize(files=DEFAULT_FILES):
    for file in files:
        filename = file['filename']
        nickname = file.get('nickname', filename)
        print('working on: ', filename)

        with h5py.File(filename + '.hdf5', 'r') as f:
            hnswparams = HNSWParams()

            hnswparams.multi = file.get('multi', False)
            hnswparams.dim = file.get('dim', len(f['train'][0][0]) if hnswparams.multi else len(f['train'][0]))
            hnswparams.metric = file['metric'] # must be set
            hnswparams.type = file.get('type', VecSimType_FLOAT32)
            hnswparams.M = file.get('M', 64)
            hnswparams.efConstruction = file.get('efConstruction', 512)
            hnswparams.initialCapacity = len(f['train'])

            metric = {VecSimMetric_Cosine: 'cosine', VecSimMetric_L2: 'euclidean'}[hnswparams.metric]
            serialized_raw_name = '%s-%s-dim%d' % (nickname, metric, hnswparams.dim)
            serialized_file_name = serialized_raw_name + '-M%d-efc%d' % (hnswparams.M, hnswparams.efConstruction)
            if hnswparams.type == VecSimType_FLOAT64:
                serialized_file_name = serialized_file_name + '-fp64'
                serialized_raw_name = serialized_raw_name + '-fp64'

            print('first, exporting test set to binary')
            if not file.get('skipRaw', False):
                test = f['test']
                if hnswparams.type == VecSimType_FLOAT64:
                    test = test.astype(np.float64)
                with open(serialized_raw_name + '-test_vectors.raw', 'wb') as testfile:
                    for vec in test:
                        testfile.write(vec.tobytes())
            else:
                print('nevermind, skipping test set export')

            print('second, indexing train set')

            index = HNSWIndex(hnswparams)

            data = f['train']
            if hnswparams.type == VecSimType_FLOAT64:
                data = data.astype(np.float64)
            for label, cur in enumerate(data):
                for vec in cur if hnswparams.multi else [cur]:
                    index.add_vector(vec, label)
                    print(label, len(data), sep='/', end='\r')
            print(len(data), len(data), sep='/')

        # Finished indexing, going back an indentation level so we close the input file

        print('lastly, serializing the index')
        output = os.path.join(location, serialized_file_name + '.hnsw_v3')
        index.save_index(output)
        print('saved to: ', output)

        # Sanity check - attempt to load the index
        index = HNSWIndex(output)

        # Release memory before next iteration
        del index

if __name__ == '__main__':
    serialize()
