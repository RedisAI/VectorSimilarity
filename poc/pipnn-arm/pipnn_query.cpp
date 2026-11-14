#include "utils/NSGDist.h"
#include "utils/euclidian_point.h"
#include "utils/point_range.h"
#include "utils/beamSearch.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace parlayANN;
int main(int argc, char** argv) {
  if (argc < 7) { std::cerr << "usage: graph base query truth k beam [beam...]\n"; return 1; }
  using id = uint32_t;
  Graph<id> graph(argv[1]);
  PointRange<Euclidian_Point<float>> base(argv[2]);
  std::ifstream statm("/proc/self/statm");
  size_t pages=0, resident=0; statm >> pages >> resident;
  struct stat graph_file{}, base_file{};
  if (!statm || stat(argv[1], &graph_file) || stat(argv[2], &base_file)) return 2;
  std::cout << "resident_index_process_bytes=" << resident * sysconf(_SC_PAGESIZE)
            << " graph_file_bytes=" << graph_file.st_size << " base_file_bytes=" << base_file.st_size << '\n';
  PointRange<Euclidian_Point<float>> queries(argv[3]);
  groundTruth<id> truth(argv[4]);
  int k = std::atoi(argv[5]);
  if (k <= 0 || queries.size() != truth.size() || k > truth.dimension() || graph.size() != base.size() || base.size() == 0) return 2;
  stats<id> qs(queries.size()); qs.clear();
  parlay::sequence<id> start = {0};
  for (int arg=6; arg<argc; ++arg) {
    int beam = std::atoi(argv[arg]);
    if (beam < k) return 2;
    QueryParams qp(k, beam, 1.35, graph.size(), graph.max_degree());
    auto run = [&](size_t i) { return beam_search_rerank(queries[i], queries[i], queries[i], graph, base, base, base, qs, start, qp); };
    for (size_t i=0; i<std::min<size_t>(100, queries.size()); ++i) run(i);
    std::vector<double> times; times.reserve(queries.size());
    std::vector<std::vector<id>> results; results.reserve(queries.size());
    for (size_t i=0; i<queries.size(); ++i) {
      auto begin = std::chrono::steady_clock::now();
      auto found = run(i);
      auto end = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double, std::micro>(end-begin).count());
      std::vector<id> row; row.reserve(found.size());
      for (auto p : found) row.push_back(p.first);
      results.push_back(std::move(row));
    }
    size_t correct=0;
    for(size_t i=0;i<queries.size();++i) {
      std::set<id> got(results[i].begin(), results[i].end());
      if (got.size()!=static_cast<size_t>(k) || *got.rbegin()>=base.size()) { std::cerr << "invalid results at " << i << '\n'; return 3; }
      for (int j=0;j<k;++j) correct += got.count(truth.coordinates(i,j));
    }
    double total=0; for (double x:times) total+=x;
    std::sort(times.begin(),times.end());
    auto pct=[&](double p){return times[static_cast<size_t>((times.size()-1)*p)];};
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::cout << "beam=" << beam << " recall=" << double(correct)/(k*queries.size()) << " p50_us=" << pct(.5) << " p95_us=" << pct(.95) << " query_qps=" << 1e6*queries.size()/total << " peak_rss_kib=" << usage.ru_maxrss << " queries=" << queries.size() << '\n';
  }
}
