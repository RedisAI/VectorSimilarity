import struct
import numpy as np
 
 
"""
                  IO Utils
"""
 
 
def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
 
def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)
 
 
def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)
 
        
def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)

if __name__ == '__main__':
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    gt_1 = read_ibin('tests/benchmark/data/deep.groundtruth.10K.10K.ibin')
    print(gt_1.shape)
    gt_2 = read_ibin('tests/benchmark/data/deep.groundtruth.100K.10K.ibin')
    print(gt_2.shape)
    # gt_3 = read_ibin('tests/benchmark/data/deep.groundtruth.public.10K.ibin')
    # print(gt_3.shape)
    vectors = read_fbin('tests/benchmark/data/deep.base.10M.fbin')
    max = np.max(np.max(vectors, axis=0))
    # vectors_100k = vectors[:100000, :]
    # write_fbin('tests/benchmark/data/deep.base.100K.fbin', vectors_100k)
    # vectors_10k = vectors[:10000, :]
    # write_fbin('tests/benchmark/data/deep.base.10K.fbin', vectors_10k)
    # groundtruth = read_ibin('tests/benchmark/data/scripts/.groundtruth_checkpoint/groundtruth_checkpoint.ibin')
    # print(groundtruth.shape)
    # pass
