import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
import pyglew as glew
import ctypes
import pycuda.driver as cuda
import pycuda.gl as cuda_gl
from pycuda.compiler import SourceModule
from pycuda import cumath
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from cudaTools import setCudaDevice, getFreeMemory
import sys, time, os

viewRangeX = 10.
viewRangeY = 10.
viewRangeZ = 10.

width_GL = 512*2
height_GL = 512*2

def dataCircle( r=1., center=(0.,0.), nPoints=100, color=(1.,1.,1.) ):
    data = [ center[0]+r, center[1] ]
    colors = [ 0., 0., 0. ]
    theta = 0.
    dTheta = 2*np.pi/(nPoints-4)
    for i in range(nPoints-3):
      data.extend([ r*np.cos(theta) +center[0], r*np.sin(theta)+center[1]])
      colors.extend( color )
      theta += dTheta
    data.extend((center[0]+r, center[1] ))
    colors.extend(color)
    data.extend((center[0]+r, center[1] ))
    colors.extend((0., 0., 0.))
    return data, colors

def circlesGrid(r, xMin, xMax, yMin, yMax, nPoints_=100, color=(1., 1., 1.)):
  col = []
  pos = []
  n = 0
  for x in range(int(xMin), int(xMax)+1):
    for y in range(int(yMin), int(yMax)+1):
      p, c = dataCircle(r, (x,y), nPoints_, color)
      pos.extend( p )
      col.extend(c)
      n += 1
  return pos, col, n

cirPos_L, cirCol_L, nCircles = circlesGrid(0.4, -5, 5, -5, 5 )
DIM = 2
nPoints = 100000
nPointsForCircles = 100*nCircles


block2D_GL = None
grid2D_GL = None

viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -4.])

GL_initialized = False 
CUDA_initialized = False 

gl_VBO = None
cuda_VOB = None
#cuda_VOB_ptr = None

frames = 0
frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

pointsPos_h = None
pointsColor = None
pointsPos_d = None 
random_d = None

updateImageDataKernel = None



def initGL():
  global GL_initialized
  if GL_initialized: return
  glutInit()
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
  glutInitWindowSize(width_GL, height_GL)
  glutInitWindowPosition(50, 50)
  glutCreateWindow("Window")
  glew.glewInit()
  glClearColor(0.0, 0.0, 0.0, 0.0)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  if width_GL <= height_GL: glOrtho(-viewRangeX/2., viewRangeX/2., 
				-viewRangeY/2.*height_GL/width_GL, viewRangeY/2.*height_GL/width_GL,
				-viewRangeZ/2., viewRangeZ/2.)
  else: glOrtho(-viewRangeX/2.*width_GL/height_GL, viewRangeX/2.*width_GL/height_GL, 
		-viewRangeY/2., viewRangeY/2.,
		-viewRangeZ/2., viewRangeZ/2.)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  GL_initialized = True
  print "OpenGL initialized"
  
def resize(w, h):
  global width_GL, height_GL
  width_GL, height_GL = w, h
  #initPixelBuffer()
  #grid2D_GL = ( iDivUp(width_GL, block2D_GL[0]), iDivUp(height_GL, block2D_GL[1]) )
  glViewport(0, 0, w, h)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  if width_GL <= height_GL: glOrtho(-viewRangeX/2., viewRangeX/2., 
				-viewRangeY/2.*height_GL/width_GL, viewRangeY/2.*height_GL/width_GL,
				-viewRangeZ/2., viewRangeZ/2.)
  else: glOrtho(-viewRangeX/2.*width_GL/height_GL, viewRangeX/2.*width_GL/height_GL, 
		-viewRangeY/2., viewRangeY/2.,
		-viewRangeZ/2., viewRangeZ/2.)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()
  
  
def createVBO():
  global gl_VBO, cuda_VOB, pointsPos_h, pointsColor, cirPos, cirCol
  gl_VBO = glGenBuffers(1)
  glBindBuffer(GL_ARRAY_BUFFER_ARB, gl_VBO)
  glBufferData(GL_ARRAY_BUFFER_ARB, pointsPos_h.nbytes + pointsColor.nbytes + cirPos.nbytes + cirCol.nbytes, None, GL_STREAM_DRAW_ARB)
  glBufferSubData(GL_ARRAY_BUFFER_ARB, 0, pointsPos_h.nbytes, (GLfloat*len(pointsPos_h))(*pointsPos_h) ); 
  glBufferSubData(GL_ARRAY_BUFFER_ARB, pointsPos_h.nbytes, pointsColor.nbytes, (GLfloat*len(pointsColor))(*pointsColor) );
  glBufferSubData(GL_ARRAY_BUFFER_ARB, pointsPos_h.nbytes+pointsColor.nbytes, cirPos.nbytes, (GLfloat*len(cirPos))(*cirPos) )
  glBufferSubData(GL_ARRAY_BUFFER_ARB, pointsPos_h.nbytes+pointsColor.nbytes+cirPos.nbytes, cirCol.nbytes, (GLfloat*len(cirCol))(*cirCol) )
  cuda_VOB = cuda_gl.RegisteredBuffer(long(gl_VBO))

def displayFunc():
  global  timer, frames
  timer = time.time()
  frames += 1
  
  #update()
 
  cuda_VOB_map = cuda_VOB.map()
  cuda_VOB_ptr, cuda_VOB_size = cuda_VOB_map.device_ptr_and_size()
  #updateImageData( cuda_VOB_ptr )
  cuda_VOB_map.unmap()


  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  #glClearColor(1., 1., 0., 1.)   #Background Color
  glPushMatrix();
  

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  
  glBindBuffer(GL_ARRAY_BUFFER_ARB, gl_VBO);
  #glBufferSubData(GL_ARRAY_BUFFER_ARB, 0, pointsPos_h.nbytes, (GLfloat*len(pointsPos_h))(*pointsPos_h) )
  
  glVertexPointer(2, GL_FLOAT, 0, ctypes.c_void_p(nPoints*(DIM+3)*4));
  glColorPointer(3, GL_FLOAT, 0,  ctypes.c_void_p(nPoints*(DIM+3)*4 + nPointsForCircles*2*4));
  glLineWidth(2.)
  glDrawArrays(GL_LINE_STRIP, 0, nPointsForCircles);
  
  
  glVertexPointer(DIM, GL_FLOAT, 0, None);
  glColorPointer(3, GL_FLOAT, 0,  ctypes.c_void_p(nPoints*DIM*4));
  glPointSize(1.)
  glDrawArrays(GL_POINTS, 0, nPoints);
    

  
  glDisableClientState(GL_VERTEX_ARRAY);  
  glDisableClientState(GL_COLOR_ARRAY);
  
  
  glBindBuffer(GL_ARRAY_BUFFER_ARB, 0);
  glPopMatrix();
  
  timer = time.time()-timer
  computeFPS()
  glutSwapBuffers();

  
 
def startGL():
  glutDisplayFunc(displayFunc)
  glutReshapeFunc(resize)
  glutIdleFunc(displayFunc)
  glutKeyboardFunc( keyboard )
  glutMouseFunc(mouse)
  #glutIdleFunc( idleFunc )
  glutMotionFunc(motion)
  glutIdleFunc(glutPostRedisplay)
  #if backgroundType == 'point': glutMotionFunc(mouseMotion_point)
  #if backgroundType == 'square': glutMotionFunc(mouseMotion_square)
  #import pycuda.autoinit
  print "Starting GLUT main loop..."
  glutMainLoop() 
  #displayFunc()

def computeFPS():
    global frameCount, fpsCount 
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle("CUDA Points 3D animation: %f fps" % ifps)
        fpsCount = 0

def keyboard(*args):
  ESCAPE = '\033'
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.Context.pop()
    sys.exit()

ox = 0
oy = 0
buttonState = 0
def mouse(button, state, x , y):
  global ox, oy, buttonState
  if state == GLUT_DOWN:
    buttonState |= 1<<button
  elif state == GLUT_UP:
    buttonState = 0
  ox = x
  oy = y
  glutPostRedisplay()

def motion(x, y):
  global viewRotation, viewTranslation
  global ox, oy, buttonState
  dx = x - ox
  dy = y - oy 
  if buttonState == 4:
    viewTranslation[2] += dy/100.
  elif buttonState == 2:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  elif buttonState == 1:
    viewRotation[0] += dy/5.
    viewRotation[1] += dx/5.
  ox = x
  oy = y
  glutPostRedisplay()


def updateImageData( cuda_ptr ):
  updateImageDataKernel( np.int32(nPoints), np.float32(viewRangeX), np.float32(-1),
			random_d, np.intp(cuda_ptr), grid=grid2D_GL, block=block2D_GL  )

def update():
  global pointsPos_h, random_d
  pointsPos_h =  2*np.random.random([nPoints*3]).astype(np.float32) -1
  random_d = curandom.rand(nPoints*3)
  
  





def startAnimation():
  if  not GL_initialized: initGL()
  if not CUDA_initialized: setCudaDevice( usingAnimation = True  )
  #initCudaGL()
  createVBO()
  startGL()


initGL()
setCudaDevice( usingAnimation = True )

#def initCudaGL():
  #global block2D_GL, grid2D_GL, updateImageDataKernel
if block2D_GL == None: block2D_GL = (512,1, 1)
if grid2D_GL == None: grid2D_GL = (nPoints/block2D_GL[0], 1, 1 )
#Read and compile CUDA code
#print "Compiling CUDA code"
########################################################################
cudaAnimCode = SourceModule('''
  #include <cuda.h>
  __global__ void updateImageData_kernel ( int N, float a, float b, float *input, float *output ){  
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    while ( tid < 3*N ){
      output[tid] = a*input[tid] + b;
      tid += blockDim.x * gridDim.x;
    }  
  }
  ''')
updateImageDataKernel = cudaAnimCode.get_function('updateImageData_kernel')
########################################################################
#global pointsPos_h, pointsColor, pointsPos_d, random_d
#Initialize all gpu data
#print "Initializing Data"
#initialMemory = getFreeMemory( show=True )  
########################################################################

pointsPos_h =  viewRangeX*np.random.random([nPoints*DIM]).astype(np.float32) -viewRangeX/2.
pointsColor = np.random.random([nPoints*3]).astype(np.float32)
cirPos = np.array(cirPos_L, dtype=np.float32)
cirCol = np.array(cirCol_L, dtype=np.float32)
########################################################################
pointsPos_d = gpuarray.to_gpu( pointsPos_h )
random_d = curandom.rand(nPoints*3)
########################################################################
#finalMemory = getFreeMemory( show=False )
#print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 











startAnimation()






