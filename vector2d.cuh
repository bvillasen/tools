//(updated: 11/28/13)
#include<cstdlib>
#include<cmath>

using namespace std;

extern "C"{
  class Vector2d_double{
  public:
    double x;
    double y;
    
    //Constructores
    __host__ __device__ Vector2d_double(){ x=0.0; y=0.0; }
    __host__ __device__ Vector2d_double(double x0, double y0) : x(x0), y(y0) {}

    __host__ __device__ double norm() { return sqrt( x*x + y*y ); }
    __host__ __device__ void normalize(){
      double mag = norm();
      x/=mag;
      y/=mag;
    }

//     __host__ __device__ void operator=(const Vector2d_double &v){
//       x = v.x;
//       y = v.y;
//     }
    __host__ __device__ Vector2d_double operator+(const Vector2d_double &v){
      return Vector2d_double( x+v.x , y+v.y);
    }
    __host__ __device__ void operator+=(const Vector2d_double &v){
      x+=v.x;
      y+=v.y;
    }
    __host__ __device__ Vector2d_double operator-(const Vector2d_double &v){
      return Vector2d_double( x-v.x , y-v.y);
    }
    __host__ __device__ void operator-=(const Vector2d_double &v){
      x-=v.x;
      y-=v.y;
    }
    __host__ __device__ double operator*(const Vector2d_double &v){
      return x*v.x + y*v.y;
    }
    //scalar mult
    __host__ __device__ Vector2d_double operator/(const double c){
      return Vector2d_double( c*x , c*y);
    }
    __host__ __device__ void operator/=(const double c){
      x *= c;
      y *= c;
    }
    
    __host__ __device__ void redefine( double x1, double y1){
      x=x1;
      y=y1;
    }
  };
}

extern "C"{
  class Vector2d_float{
  public:
    float x;
    float y;
    
    //Constructores
    __host__ __device__ Vector2d_float(){ x=0.0; y=0.0; }
    __host__ __device__ Vector2d_float(float x0, float y0) : x(x0), y(y0) {}

    __host__ __device__ float norm() { return sqrt( x*x + y*y ); }
    __host__ __device__ void normalize(){
      float mag = norm();
      x/=mag;
      y/=mag;
    }
    
    __host__ __device__ Vector2d_float operator+(const Vector2d_float &v){
      return Vector2d_float( x+v.x , y+v.y);
    }
    __host__ __device__ void operator+=(const Vector2d_float &v){
      x+=v.x;
      y+=v.y;
    }
    __host__ __device__ Vector2d_float operator-(const Vector2d_float &v){
      return Vector2d_float( x-v.x , y-v.y);
    }
    __host__ __device__ void operator-=(const Vector2d_float &v){
      x-=v.x;
      y-=v.y;
    }
    __host__ __device__ float operator*(const Vector2d_float &v){
      return x*v.x + y*v.y;
    }
    //scalar mult
    __host__ __device__ Vector2d_float operator/(const float c){
      return Vector2d_float( c*x , c*y);
    }
    __host__ __device__ void operator/=(const float c){
      x *= c;
      y *= c;
    }
    
    __host__ __device__ void redefine( float x1, float y1){
      x=x1;
      y=y1;
    }
  };
}