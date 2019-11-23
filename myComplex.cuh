#include <iostream>
#include <math.h>
using namespace std;



//Clase complejo que se puede llamar desde device y host
class myComplex {
public:  
  float   r;
  float   i;
  //Constructor
  __host__ __device__ myComplex( float a=0, float b=0 ) : r(a), i(b)  {}
  
  __host__ __device__ float norm( void ) {
      return r * r + i * i;
  }
  __host__ __device__ myComplex conj( void ) {
      return myComplex(r, -i);
  }
  __host__ __device__ myComplex operator*(const myComplex& a) {
      return myComplex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
    __host__ __device__ myComplex operator*(const float a) {
      return myComplex(r*a, i*a);
  }
 __host__ __device__ myComplex operator*(const int a) {
      return myComplex(r*a, i*a);
  }

  __host__ __device__ myComplex operator+(const myComplex& a) {
      return myComplex(r+a.r, i+a.i);
  }
   __host__ __device__ myComplex operator-(const myComplex& a) {
      return myComplex(r-a.r, i-a.i);
  }

};


__host__ __device__ myComplex exp(const myComplex& b ) {
      return myComplex(exp(b.r)*cos(b.i),exp(b.r)*sin(b.i));
  }
  