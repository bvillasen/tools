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
#from pycuda import cumath
import pycuda.gpuarray as gpuarray
#import pycuda.curandom as curandom
from cudaTools import setCudaDevice, getFreeMemory
import sys, time, os
import pylab as plt

viewXmin, viewXmax = None, None
viewYmin, viewYmax = None, None
viewZmin, viewZmax = None, None


width_GL = 512*2
height_GL = 512*2

DIM = 2
nPoints = None
nPointsForCircles = None


block_GL = None
grid_GL = None

viewRotation =  np.zeros(3).astype(np.float32)
viewTranslation = np.array([0., 0., -4.])

GL_initialized = False 
CUDA_initialized = False 

gl_VBO = None
cuda_VOB = None
cuda_VOB_ptr = None

frames = 0
frameCount = 0
fpsCount = 0
fpsLimit = 8
timer = 0.0

gridCenter = (0,0)
showGrid = True
nCirclesGrid = None
nPointsPerCircle = None
cirPos = None
cirCol = None
pointsPos_h = None
pointsColor = None
pointsPos_d = None 
random_d = None

updateImageDataKernel = None

def initGL():
  global GL_initialized
  if GL_initialized: return
  glutInit()
  glew.glewInit()
  GL_initialized = True
  print "OpenGL initialized"
  #openGLWindow()
  glutInitWindowSize(width_GL, height_GL)
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
  glutInitWindowPosition(0, 0)
  glutCreateWindow("Window")
  
def openGLWindow():  
  glClearColor(0.0, 0.0, 0.0, 0.0)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  if width_GL <= height_GL: glOrtho( viewXmin, viewXmax, 
				viewYmin*height_GL/width_GL, viewYmax*height_GL/width_GL,
				viewZmin, viewZmax)
  else: glOrtho(viewXmin*width_GL/height_GL, viewXmax*width_GL/height_GL, 
		viewYmin, viewYmax,
		viewZmin, viewZmax)
  glMatrixMode(GL_MODELVIEW)
  glLoadIdentity()  
  
  
def resize(w, h):
  global width_GL, height_GL
  width_GL, height_GL = w, h
  #initPixelBuffer()
  #grid_GL = ( iDivUp(width_GL, block_GL[0]), iDivUp(height_GL, block_GL[1]) )
  glViewport(0, 0, w, h)
  glMatrixMode(GL_PROJECTION)
  glLoadIdentity()
  if width_GL <= height_GL: glOrtho( viewXmin, viewXmax, 
				viewYmin*height_GL/width_GL, viewYmax*height_GL/width_GL,
				viewZmin, viewZmax)
  else: glOrtho(viewXmin*width_GL/height_GL, viewXmax*width_GL/height_GL, 
		viewYmin, viewYmax,
		viewZmin, viewZmax)
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
  global  timer, frames, cuda_VOB_ptr
  timer = time.time()
  frames += 1
  
  cuda_VOB_map = cuda_VOB.map()
  cuda_VOB_ptr, cuda_VOB_size = cuda_VOB_map.device_ptr_and_size()
  updateFunc()
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
  if showGrid: glDrawArrays(GL_LINE_STRIP, 0, nPointsForCircles);
  
  glVertexPointer(DIM, GL_FLOAT, 0, None);
  glColorPointer(3, GL_FLOAT, 0,  ctypes.c_void_p(nPoints*DIM*4));
  glPointSize(1.5)
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
  glutSpecialFunc(specialKeys)
  glutMouseFunc(mouse)
  #glutMouseWheelFunc(mouseWheel)
  glutIdleFunc( displayFunc )
  glutMotionFunc(motion)
  glutIdleFunc(glutPostRedisplay)
  #if backgroundType == 'point': glutMotionFunc(mouseMotion_point)
  #if backgroundType == 'square': glutMotionFunc(mouseMotion_square)
  print "\nStarting GLUT main loop..."
  glutMainLoop() 
  #displayFunc()

def computeFPS():
    global frameCount, fpsCount 
    frameCount += 1
    fpsCount += 1
    if fpsCount == fpsLimit:
        ifps = 1.0 /timer
        glutSetWindowTitle("CUDA Points 2D animation: %f fps" % ifps)
        fpsCount = 0

def keyboard(*args):
  global viewXmin, viewXmax, viewYmin, viewYmax
  global showGrid, gridCenter
  ESCAPE = '\033'
  if args[0] == ESCAPE:
    print "Ending Simulation"
    #cuda.Context.pop()
    sys.exit()
  if args[0] == " ":
    viewXmin = -2500
    viewXmax = 2500
    viewYmin = -2500
    viewYmax = 2500
    resize( width_GL, height_GL )
  if args[0] == "g":
    showGrid = not showGrid
    if showGrid:
      moveGrid(int(viewXmax+viewXmin)/2-gridCenter[0],int(viewYmax+viewYmin)/2-gridCenter[1])
      gridCenter = (int(viewXmax+viewXmin)/2, int(viewYmax+viewYmin)/2)
  if args[0] == "p":
    plt.figure(0)
    plt.clf()
    plt.plot(np.ones(100))  
    plt.show()

ox = 0
oy = 0
buttonState = 0
def mouse(button, state, x , y):
  global ox, oy, buttonState
  global viewXmax, viewXmin, viewYmax, viewYmin
  zoomFactor = 1.5
  zoom = "center"
  if button == 3:  #wheel up
    rangeX = viewXmax-viewXmin
    rangeY = viewYmax-viewYmin
    if zoom == "pointer":
      pointerX = rangeX*x/width_GL + viewXmin
      pointerY = rangeY*(height_GL-y)/height_GL + viewYmin
    elif zoom == "center":
      pointerX = rangeX*0.5 + viewXmin
      pointerY = rangeY*0.5 + viewYmin
    viewXmin = pointerX - rangeX/(2.*zoomFactor)
    viewXmax = pointerX + rangeX/(2*zoomFactor)
    viewYmin = pointerY + rangeY/(2*zoomFactor)
    viewYmax = pointerY - rangeY/(2*zoomFactor)
    resize(width_GL, height_GL) 
  if button == 4:  #wheel down  
    rangeX = viewXmax-viewXmin
    rangeY = viewYmax-viewYmin
    if zoom == "pointer":
      pointerX = rangeX*x/width_GL + viewXmin
      pointerY = rangeY*(height_GL-y)/height_GL + viewYmin
    elif zoom == "center":
      pointerX = rangeX*0.5 + viewXmin
      pointerY = rangeY*0.5 + viewYmin
    viewXmin = pointerX - rangeX/2.*zoomFactor
    viewXmax = pointerX + rangeX/2.*zoomFactor
    viewYmin = pointerY + rangeY/2.*zoomFactor
    viewYmax = pointerY - rangeY/2.*zoomFactor
    resize(width_GL, height_GL)  
  elif state == GLUT_DOWN:
    buttonState |= 1<<button
  elif state == GLUT_UP:
    buttonState = 0
  ox = x
  oy = y
  glutPostRedisplay()

def motion(x, y):
  global viewXmax, viewXmin, viewYmax, viewYmin
  global ox, oy, buttonState
  dx = x - ox
  dy = y - oy 
  if buttonState == 1:
    move = (viewXmax-viewXmin)/1000.
    viewXmin -= dx*move
    viewXmax -= dx*move
    viewYmin += dy*move
    viewYmax += dy*move
    resize( width_GL, height_GL )
  elif buttonState == 4:
    viewTranslation[2] += dy/100.
  elif buttonState == 2:
    viewTranslation[0] += dx/100.
    viewTranslation[1] -= dy/100.
  ox = x
  oy = y
  glutPostRedisplay()

def specialKeys( key, x, y ):
  global viewXmin, viewXmax, viewYmin, viewYmax
  if key==GLUT_KEY_UP:
    resize(width_GL, height_GL)
  if key==GLUT_KEY_DOWN:
    resize(width_GL, height_GL)
    
def updateFunc():
  print "Default update function"

def initCudaGL():
  global block_GL, grid_GL, updateImageDataKernel
  if block_GL == None: block_GL = (512,1, 1)
  if grid_GL == None: grid_GL = (nPoints/block_GL[0], 1, 1 )
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
  global pointsPos_h, pointsColor, pointsPos_d, random_d, cirPos, cirCol
  #Initialize all gpu data
  #print "Initializing Data"
  #initialMemory = getFreeMemory( show=True )  
  ########################################################################
  pointsPos_h = np.random.random([nPoints*DIM]).astype(np.float32) - 1.
  pointsColor = np.random.random([nPoints*3]).astype(np.float32)
  ########################################################################
  #finalMemory = getFreeMemory( show=False )
  #print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 

def moveGrid(dx, dy):
  cirPos[::2] += np.float32(dx)
  cirPos[1::2] += np.float32(dy)
  glBindBuffer(GL_ARRAY_BUFFER_ARB, gl_VBO)
  glBufferSubData(GL_ARRAY_BUFFER_ARB, pointsPos_h.nbytes+pointsColor.nbytes, cirPos.nbytes, (GLfloat*len(cirPos))(*cirPos) )

def startAnimation():
  global nPointsForCircles
  if  not GL_initialized: initGL()
  openGLWindow()
  if not CUDA_initialized: setCudaDevice( usingAnimation = True  )
  nPointsForCircles = nPointsPerCircle*nCirclesGrid
  initCudaGL()
  createVBO()
  startGL()


#initGL()
#setCudaDevice( usingAnimation = True )



#cirPos, cirCol, nCirclesGrid = circlesGrid( 0.4, -5., 5., -5., 5.)

#startAnimation()






