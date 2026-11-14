#include <VecSim/vec_sim.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
template<class T> struct Matrix { uint32_t n, d; std::vector<T> v; };
template<class T> Matrix<T> read_matrix(const char *path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("open: ") + path);
    auto size = f.tellg(); f.seekg(0); Matrix<T> a{};
    f.read(reinterpret_cast<char*>(&a.n), 4); f.read(reinterpret_cast<char*>(&a.d), 4);
    uint64_t count = uint64_t(a.n)*a.d;
    if (!f || !a.n || !a.d || uint64_t(size) != 8 + count*sizeof(T)) throw std::runtime_error("invalid matrix size/header");
    a.v.resize(count); f.read(reinterpret_cast<char*>(a.v.data()), count*sizeof(T));
    if (!f) throw std::runtime_error("short matrix read"); return a;
}
long rss_bytes() { std::ifstream f("/proc/self/statm"); long total, resident; f >> total >> resident; return resident * sysconf(_SC_PAGESIZE); }
long peak_bytes() { rusage r{}; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss * 1024L; }
int main(int argc, char **argv) { try {
    auto program_start=Clock::now();
    if (argc < 8) throw std::runtime_error("usage: vecsim_bench base.fbin query.fbin gt.ibin M efConstruction repeat ef[,ef...] [query_limit]");
    auto base=read_matrix<float>(argv[1]);
    if (base.n<10) throw std::runtime_error("base must contain at least 10 vectors");
    size_t M=std::stoul(argv[4]), efc=std::stoul(argv[5]); int repeat=std::stoi(argv[6]);
    if (!M || !efc) throw std::runtime_error("invalid arguments");
    auto read_done=Clock::now(); long rss_before=rss_bytes();
    VecSimParams p{}; p.algo=VecSimAlgo_HNSWLIB; auto &h=p.algoParams.hnswParams;
    h.type=VecSimType_FLOAT32; h.dim=base.d; h.metric=VecSimMetric_L2; h.M=M; h.efConstruction=efc;
    auto build_start=Clock::now(); auto index=VecSimIndex_New(&p); if (!index) throw std::runtime_error("index creation failed");
    auto start=Clock::now();
    for(size_t i=0;i<base.n;i++) if(VecSimIndex_AddVector(index,base.v.data()+i*base.d,i)!=1) throw std::runtime_error("insert failed");
    auto build_end=Clock::now(); double build=std::chrono::duration<double>(build_end-build_start).count(); double insert=std::chrono::duration<double>(build_end-start).count();
    auto build_peak=peak_bytes(); auto stats=VecSimIndex_StatsInfo(index);
    auto base_n=base.n, base_d=base.d;
    std::vector<float>().swap(base.v); malloc_trim(0);
    auto resident_index_process=rss_bytes();
    std::cout << "n=" << base_n << " dim=" << base_d << " M=" << M << " efConstruction=" << efc << " seed=100(fixed_C_API) repeat=" << repeat << " base_read_prepare_seconds=" << std::chrono::duration<double>(read_done-program_start).count() << " build_seconds=" << build << " insert_seconds=" << insert << " ready_from_fbin_seconds=" << std::chrono::duration<double>(build_end-program_start).count() << " rss_before_build=" << rss_before << " index_bytes=" << stats.memory << " resident_index_process_bytes=" << resident_index_process << " build_peak_rss_bytes=" << build_peak << '\n';
    auto queries=read_matrix<float>(argv[2]); auto gt=read_matrix<uint32_t>(argv[3]);
    if (base_d!=queries.d || gt.n!=queries.n || gt.d<10) throw std::runtime_error("incompatible dimensions");
    for (auto id:gt.v) if(id>=base_n) throw std::runtime_error("ground truth ID out of range");
    size_t nq=argc>8 ? std::min<size_t>(queries.n,std::stoul(argv[8])) : queries.n;
    if (!nq) throw std::runtime_error("invalid query count");
    std::cout << "queries=" << nq << '\n';
    std::string efs=argv[7]; size_t pos=0;
    while(pos<efs.size()) { size_t next=efs.find(',',pos); size_t ef=std::stoul(efs.substr(pos,next-pos)); if(ef<10) throw std::runtime_error("ef must be >= 10");
      VecSimQueryParams qp{}; qp.hnswRuntimeParams.efRuntime=ef; size_t hits=0; std::vector<double> latency; latency.reserve(nq);
      for(size_t q=0;q<std::min<size_t>(nq,100);q++) { auto warm=VecSimIndex_TopKQuery(index,queries.v.data()+q*queries.d,10,&qp,BY_SCORE); if(!warm || VecSimQueryReply_GetCode(warm)!=VecSim_QueryReply_OK) throw std::runtime_error("warmup failed"); VecSimQueryReply_Free(warm); }
      auto sweep_start=Clock::now();
      for(size_t q=0;q<nq;q++) { auto t=Clock::now(); auto reply=VecSimIndex_TopKQuery(index,queries.v.data()+q*queries.d,10,&qp,BY_SCORE); auto end=Clock::now();
        if(!reply || VecSimQueryReply_GetCode(reply)!=VecSim_QueryReply_OK || VecSimQueryReply_Len(reply)!=10) throw std::runtime_error("query failed");
        latency.push_back(std::chrono::duration<double,std::micro>(end-t).count()); auto it=VecSimQueryReply_GetIterator(reply);
        std::vector<int64_t> seen; while(VecSimQueryReply_IteratorHasNext(it)) { auto id=VecSimQueryResult_GetId(VecSimQueryReply_IteratorNext(it)); if(std::find(seen.begin(),seen.end(),id)!=seen.end()) throw std::runtime_error("duplicate result ID"); seen.push_back(id); for(size_t j=0;j<10;j++) if(id==gt.v[q*gt.d+j]) { hits++; break; } }
        VecSimQueryReply_IteratorFree(it); VecSimQueryReply_Free(reply);
      }
      double elapsed=std::chrono::duration<double>(Clock::now()-sweep_start).count(); double query_us=std::accumulate(latency.begin(),latency.end(),0.0); std::sort(latency.begin(),latency.end());
      std::cout << "ef=" << ef << " recall10=" << double(hits)/(10*nq) << " p50_us=" << latency[(nq-1)*50/100] << " p95_us=" << latency[(nq-1)*95/100] << " serial_query_qps=" << 1e6*nq/query_us << " sweep_qps_including_recall=" << nq/elapsed << '\n';
      if(next==std::string::npos) break; pos=next+1;
    }
    VecSimIndex_Free(index); return 0;
} catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; } }
