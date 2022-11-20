# Copyright Redis Ltd. 2021 - present
# Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
# the Server Side Public License v1 (SSPLv1).

import redis

from functools import wraps

'''
python -m RLTest --test test_vecsim_malloc.py --module path/to/memory_test.so
'''

def with_test_module(f):
    @wraps(f)
    def wrapper(env, *args, **kwargs):
        con = env.getConnection()
        modules = con.execute_command("MODULE", "LIST")
        if b'VecSim_memory' in [module[1] for module in modules]:
            return f(env, *args, **kwargs)
        try:
            ret = con.execute_command('MODULE', 'LOAD', 'memory_test.so')
            env.assertEqual(ret, b'OK')
        except Exception as e:
            env.assertFalse(True)
            env.debugPrint(str(e), force=True)
            return
        return f(env, *args, **kwargs)
    return wrapper

@with_test_module
def test_create_bf_and_hnsw_indexes_check(env):

    con = env.getConnection()

    ret = con.execute_command('VecSim_memory.create_index_check', 'BF')
    env.assertEqual(ret, b'OK')
    ret = con.execute_command('VecSim_memory.create_index_check', 'HNSW')
    env.assertEqual(ret, b'OK')

@with_test_module
def test_create_index_add_check_bf(env):

    con = env.getConnection()

    for n in range(1, 1000000, 50000):
        ret = con.execute_command('VecSim_memory.create_index_add_n_check', 'BF', n)
        env.assertEqual(ret, b'OK')
    
@with_test_module
def test_create_index_add_check_hnsw(env):

    con = env.getConnection()
    vectors = [1000, 100000]

    for n in vectors:
        ret = con.execute_command('VecSim_memory.create_index_add_n_check', 'HNSW', n)
        env.assertEqual(ret, b'OK')
 
@with_test_module
def test_create_index_add_del_check_bf(env):

    con = env.getConnection()

    for n in range(1, 1000000, 50000):
        ret = con.execute_command('VecSim_memory.create_index_add_n_delete_m_check', 'BF', n, int(n/2))
        env.assertEqual(ret, b'OK')
        ret = con.execute_command('VecSim_memory.create_index_add_n_delete_m_check', 'BF', n, n)
        env.assertEqual(ret, b'OK')
    
@with_test_module
def test_create_index_add_del_check_hnsw(env):

    con = env.getConnection()
    vectors = [1000, 100000]

    for n in vectors:
        ret = con.execute_command('VecSim_memory.create_index_add_n_delete_m_check', 'HNSW', n, n)
        env.assertEqual(ret, b'OK')
