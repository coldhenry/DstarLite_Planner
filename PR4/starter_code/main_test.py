# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:44:08 2020

@author: coldhenry
"""

import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
#from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#import Planner
import Weighted_Astar_Planner as Astar
from Quadrotor2 import Quadrotor2
import matplotlib.animation as animation


show_animation = True
x_pos = -5
y_pos = -5
z_pos = 5
x_vel = 0
y_vel = 0
z_vel = 0
x_acc = 0
y_acc = 0
z_acc = 0
roll = 0
pitch = 0
yaw = 0
roll_vel = 0
pitch_vel = 0
yaw_vel = 0

des_yaw = 0

dt = 0.1
t = 0

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  

def load_map(fname):
  '''
  Loads the bounady and blocks from map file fname.
  
  boundary = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  
  blocks = [['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'],
            ...,
            ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']]
  '''
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  '''
  Visualization of a planning problem with environment boundary, obstacle blocks, and start and goal points
  '''
  #plt.style.use('dark_background')
  fig = plt.figure()
  ax = p3.Axes3D(fig)
  #ax = fig.add_subplot(111, projection='3d')
  
  fig.set_facecolor('black')
  ax.set_facecolor('black')
  ax.grid(False)
  ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
  ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
  ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
  
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
  
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])  
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  '''
  Subroutine used by draw_map() to display the environment blocks
  '''
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def update_lines(num, lines, datas):
    # NOTE: there is no .set_data() for 3 dim data...
    
    lines[0].set_data(datas[0][0:2, :num])
    lines[0].set_3d_properties(datas[0][2, :num])
    
    lines[1].set_data(datas[1][num][0:2])
    lines[1].set_3d_properties(datas[1][num][2])
    
    lines[2].set_data(datas[2][num][0:2])
    lines[2].set_3d_properties(datas[2][num][2])
    
    lines[3].set_data(datas[3][num][0:2])
    lines[3].set_3d_properties(datas[3][num][2])
    return lines

def runtest(mapfile, start, goal, verbose = True):
  '''
  This function:
   * load the provided mapfile
   * creates a motion planner
   * plans a path from start to goal
   * checks whether the path is collision free and reaches the goal
   * computes the path length as a sum of the Euclidean norm of the path segments
  '''
  # Load a map and instantiate a motion planner
  boundary, blocks = load_map(mapfile)
  print("boundry")
  print(boundary)
  print("blocks")
  print(blocks)
  MP = Astar.Weighted_Astar_Planner(blocks, boundary) # TODO: replace this with your own planner implementation
  
  # # Display the environment
  # if verbose:
    
  
  # Call the motion planner
  t0 = tic()
  path = MP.planning(start, goal)
  path = np.transpose(path)
  toc(t0,"Planning")
  #print(path)
  
  iteration = path.shape[1]
  
# data of sticks and points
  set_pts, set_st1, set_st2 = [],[],[]
  q = Quadrotor2(x=x_pos, y=y_pos, z=z_pos, roll=roll,
                     pitch=pitch, yaw=yaw, size=1, show_animation=show_animation)
  for i in range(iteration):
      q.update_pose(path[0,:][i], path[1,:][i], path[2,:][i], roll, pitch, yaw)
      set_pts.append(np.array(q.set_pts))
      set_st1.append(np.array(q.set_st1))
      set_st2.append(np.array(q.set_st2))
  
  # Fifty lines of random 3-D lines
  # data type: trajectory size: np.array((iteration, dim))
  #            set of sticks size: list of np.array 
  
  #data = [Gen_RandLine(25, 3)]
  datas = [path]
  datas.append(set_st1)
  datas.append(set_st2)
  datas.append(set_pts)
  
  
  # Plot the path
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    
    # Creating line objects.
    # NOTE: Can't pass empty arrays into 3d version of plot()
    # ax.plot(x,y,z) x,y,z can put single value or list of coordinates
    lines = [ax.plot(datas[0][0, 0:1], datas[0][1, 0:1], datas[0][2, 0:1],'y:')[0]]
    lines.append(ax.plot(set_st1[0][0],set_st1[0][1],set_st1[0][2],'r-')[0])
    lines.append(ax.plot(set_st2[0][0],set_st2[0][1],set_st2[0][2],'r-')[0])
    lines.append(ax.plot(set_pts[0][0],set_pts[0][1],set_pts[0][2],'w.')[0])
    
    #ax.plot(path[0,:],path[1,:],path[2,:],'r.')
    
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, iteration, fargs=(lines, datas),
                                       interval=50, blit=False)
    line_ani.save('basic_animation_window.gif', fps=15, extra_args=['-vcodec', 'libx264'])
    
    plt.ioff()
    plt.show()

  # TODO: You should verify whether the path actually intersects any of the obstacles in continuous space
  # TODO: You can implement your own algorithm or use an existing library for segment and 
  #       axis-aligned bounding box (AABB) intersection 
  collision = False
  goal_reached = sum((path[:,-1]-goal)**2) <= 0.25
  success = (not collision) and goal_reached
  pathlength = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
  return success, pathlength


def test_single_cube(verbose = False):
  print('Running single cube test...\n') 
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 5.5])
  success, pathlength = runtest('./maps/single_cube.txt', start, goal, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')
  
  
def test_maze(verbose = False):
  print('Running maze test...\n') 
  start = np.array([0.0, 0.0, 1.0])
  goal = np.array([12.0, 12.0, 5.0])
  success, pathlength = runtest('./maps/maze.txt', start, goal, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

    
def test_window(verbose = False):
  print('Running window test...\n') 
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([6.0, 18.0, 3.0])
  success, pathlength = runtest('./maps/window.txt', start, goal, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

  
def test_tower(verbose = False):
  print('Running tower test...\n') 
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  success, pathlength = runtest('./maps/tower.txt', start, goal, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')

     
def test_flappy_bird(verbose = False):
  print('Running flappy bird test...\n') 
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  runtest('./maps/flappy_bird.txt', start, goal, verbose) #success, pathlength = 
  # print('Success: %r'%success)
  # print('Path length: %d'%pathlength) 
  # print('\n')

  
def test_room(verbose = False):
  print('Running room test...\n') 
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  success, pathlength = runtest('./maps/room.txt', start, goal, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


def test_monza(verbose = False):
  print('Running monza test...\n')
  start = np.array([0.5, 1.0, 4.9])
  goal = np.array([3.8, 1.0, 0.1])
  success, pathlength = runtest('./maps/monza.txt', start, goal, verbose)
  print('Success: %r'%success)
  print('Path length: %d'%pathlength)
  print('\n')


if __name__=="__main__":
    

  #test_single_cube(True)
  #test_maze(True)
  #test_flappy_bird(True)
  #test_monza(True)
  test_window(True)
  #test_tower(True)
  #test_room(True)








