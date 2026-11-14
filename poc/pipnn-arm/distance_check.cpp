#include "upstream/algorithms/utils/NSGDist.h"
#include <cmath>
#include <cstdio>
int main() {
  efanna2e::DistanceL2 l2; efanna2e::DistanceInnerProduct dot; efanna2e::DistanceFastL2 norm;
  float a[100], b[100];
  for (int i=0;i<100;i++) { a[i]=(i%7-3)*0.125f; b[i]=(i%11-5)*0.25f; }
  for (unsigned n=1;n<=100;n++) {
    double l=0,d=0,z=0;
    for(unsigned i=0;i<n;i++){double x=a[i]-b[i];l+=x*x;d+=a[i]*b[i];z+=a[i]*a[i];}
    if(std::abs(l2.compare(a,b,n)-l)>1e-5 || std::abs(dot.compare(a,b,n)-d)>1e-5 || std::abs(norm.norm(a,n)-z)>1e-5) return 1;
  }
  std::puts("PASS L2, dot, norm dimensions 1..100");
}
