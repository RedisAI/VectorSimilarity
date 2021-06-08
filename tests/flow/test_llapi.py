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
