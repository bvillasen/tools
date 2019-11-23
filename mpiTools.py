from mpi4py import MPI
import numpy as np
import time

def get_neighbors_ids_2D( pId_x, pId_y, nP_x, nP_y):
  l_id = nP_x-1 if pId_x==0 else pId_x - 1
  r_id = 0 if pId_x==nP_x-1 else pId_x + 1
  d_id = nP_y-1 if pId_y==0 else pId_y - 1
  u_id = 0 if pId_y==nP_y-1 else pId_y + 1
  pId_l = l_id + pId_y*nP_x
  pId_r = r_id + pId_y*nP_x
  pId_d = pId_x + d_id*nP_x
  pId_u = pId_x + u_id*nP_x
  return pId_l, pId_r, pId_d, pId_u


def mpi_transferData_oneWay( pId, sourceId, destId, dataSend, dataReceive, tag, MPIcomm, block=True ):
  if sourceId == destId: return
  if pId == sourceId:
    request = MPIcomm.Isend(dataSend, dest=destId, tag=tag)
  if pId == destId:
    request = MPIcomm.Irecv(dataReceive, source=sourceId, tag=tag)
  # if block: MPIcomm.Barrier()
  return request

def transferData( pId, dest, source, dataOut, dataIn, tag, MPIcomm, block=True ):
  if pId%2 == 0:
    MPIcomm.Send(dataOut, dest=dest, tag=tag)
    MPIcomm.Recv(dataIn, source=source, tag=tag)
  else:
    MPIcomm.Recv(dataIn, source=source, tag=tag)
    MPIcomm.Send(dataOut, dest=dest, tag=tag)
  if block: MPIcomm.Barrier()



def get_mpi_id_2D( pId, nP_x ):
  pId_x = pId % nP_x  #peocess id in the x axis
  pId_y = pId / nP_x  #peocess id in the y axis
  return pId_x, pId_y

def get_mpi_id_3D( pId, nProc_x, nProc_y ):
  pId_x = pId % nProc_x
  pId_z = pId // (nProc_x*nProc_y)
  pId_y = (pId-pId_z*nProc_y*nProc_x) // nProc_x
  return ( pId_z, pId_y, pId_x )

def print_mpi( string , pId, nProcess, MPIcomm, secs=0.1 ):
  for i in range(nProcess):
    if pId == i:
      print '[pId {0}] '.format(pId) + string
      time.sleep(0.1)
    MPIcomm.Barrier()
