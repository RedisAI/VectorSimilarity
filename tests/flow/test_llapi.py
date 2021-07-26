import redis
import sys

from RLTest import Env
from RLTest import Defaults

if sys.version_info > (3, 0):
    Defaults.decode_responses = True

def test_llapi_hnswlib_vector_add(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_vector_add")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_search(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_search")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_search_order_by_id(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_search_order_by_id")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_search_million(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_search_million")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_indexing_same_vector(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_indexing_same_vector")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_reindexing_same_vector(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_reindexing_same_vector")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_reindexing_same_vector_different_id(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_reindexing_same_vector_different_id")
    env.assertEquals(res, "OK")

def test_llapi_hnswlib_sanity_rinsert_1280(env):
    con = env.getConnection()
    res = con.execute_command("vec_sim_test.hnswlib_sanity_rinsert_1280")
    env.assertEquals(res, "OK")