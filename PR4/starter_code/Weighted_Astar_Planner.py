# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:54:00 2020

@author: coldhenry
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Weighted_Astar_Planner:
    
    
    def __init__(self, blocks, boundry, eps=2, rr=0.1, reso=0.25):
        '''
        Grid map initialization
        '''
        self.rr = rr
        self.reso = reso
        self.get_obstacle_map(blocks,boundry)
        self.eps = eps
        
    class Node:
        
        def __init__(self, x, y, z, cost, pind):
            # grid position of node
            self.x = x
            self.y = y
            self.z = z
            self.cost = cost
            self.pind = pind # parent's grid number           
            
        def printNode(self):
            return (str(self.x) + "," + str(self.y) + "," + str(self.z))
            
        
    def planning(self, start, goal):
        """
        Weighted A* algorithm
        """
        print("start: {s}, goal: {g}".format(s=start, g=goal))
        sx, sy, sz = start[0], start[1], start[2]
        gx, gy, gz = goal[0], goal[1], goal[2]
        # change to grid index
        s_x, s_y, s_z = self.length2grid(sx, self.minx),self.length2grid(sy, self.miny),self.length2grid(sz, self.minz)
        g_x, g_y, g_z = self.length2grid(gx, self.minx),self.length2grid(gy, self.miny),self.length2grid(gz, self.minz)
        print("start point: {a},{b},{c}".format(a=s_x,b=s_y,c=s_z))        
        print("goal point: {a},{b},{c}".format(a=g_x,b=g_y,c=g_z)) 
        
        # create nodes for the start and the goal
        n_start = self.Node(s_x, s_y, s_z, 0.0, -1)
        n_goal = self.Node(g_x, g_y, g_z, 0.0, -1)

        # create the OPEN and CLOSED
        # put the start point into OPEN
        # OPEN set: key: grid number / value: node
        open_set, closed_set = dict(), dict()
        open_set[self.get_grid_number(n_start)] = n_start
        goal_id = self.get_grid_number(n_goal)
        print("goal id",goal_id)

        count = 0
        # iterate unitl the goal has been explored
        while 1:
            
            if len(open_set) == 0:
                print("OPEN set is EMPTY")
                break
            
            if goal_id in closed_set:
                print("Goal has explored, iteration used: {c}".format(c=count))
                n_goal.cost = closed_set[goal_id].cost
                n_goal.pind = closed_set[goal_id].pind
                break
            
            count += 1
            if count%1000==0:
                print("iterations: ",count)
            
            # remove node with smallest f_i (g_i+eps*h_i), call it node i
            # note: o in open_set[o] is grid number
            c_id = min(open_set, key= lambda o: open_set[o].cost+self.eps*self.heuristic(open_set[o], n_goal))
            current = open_set[c_id]
    
            del open_set[c_id]
            
            # insert node i into CLOSED
            # CLOSED set: key: grid number / value: node
            closed_set[c_id] = current
            
            # looking for valid node j
            for motion in self.expand():
                
                dx, dy, dz, c_ij = motion[0], motion[1], motion[2], motion[3]
                                                
                node_j = self.Node(current.x+dx, current.y+dy, current.z+dz, current.cost+c_ij, c_id)
                j_id = self.get_grid_number(node_j)
                
                if node_j.x == n_goal.x and node_j.y == n_goal.y and node_j.z == n_goal.z:
                    print("Find goal")
                
                # if node is in a valid position or it's already in CLOSED set
                if not self.valid_node(node_j) or j_id in closed_set.keys():
                    continue
                
                # if it doesnt exist in OPEN, add it
                if j_id not in open_set.keys():
                    #print("new node:",node_j.printNode())
                    open_set[j_id] = node_j

                # otherwise, update it
                else:
                    if open_set[j_id].cost > node_j.cost:
                        #print("update node:",node_j.printNode())
                        open_set[j_id] = node_j
            
        # derive the whole path
        path = self.get_final_path(n_goal, closed_set)
 
        return path
   
        
    def get_final_path(self, n_goal, closed_set):
        node_g = [self.grid2length(n_goal.x,self.minx),self.grid2length(n_goal.y,self.miny),self.grid2length(n_goal.z,self.minz)]
        path = [node_g]
        pind = n_goal.pind
        count = 0
        # start from goal, keep running if havent met the start
        while pind != -1:
            count += 1
            n = closed_set[pind]
            cx = self.grid2length(n.x, self.minx)
            cy = self.grid2length(n.y, self.miny)
            cz = self.grid2length(n.z, self.minz)
            path.append([cx,cy,cz])
            pind = n.pind
            if count > 5000:
                print("Path generation failed")
                break
        
        print("Path generated")
        return np.array(path)
        
        
    @staticmethod
    def heuristic(n1,n2):
        w = 1.0 #weight
        d = w*(abs(n1.x-n2.x)+ abs(n1.y-n2.y)+ abs(n1.z-n2.z) ) #+ (n1.z-n2.z)
        #d = w*np.sqrt((n1.x-n2.x)**2+ (n1.y-n2.y)**2+ (n1.z-n2.z)**2)
        #print("d",d)
        return d
    
    def get_grid_number(self, node):
        return ((node.y-self.miny) * self.xwidth + (node.x-self.minx) + self.xwidth* self.ywidth* (node.z-self.minz)).astype(int)
             
    def grid2length(self, index, min_pos):       
        original_length = index* self.reso + min_pos
        return original_length    
    
    def length2grid(self, pos, min_p):
        grid = (round((pos-min_p) / self.reso)).astype(int)        
        return grid
           
    def valid_node(self, node):
        px = self.grid2length(node.x, self.minx)
        py = self.grid2length(node.y, self.miny)
        pz = self.grid2length(node.z, self.minz)
        
        # possible node check
        if px < self.minx or px >= self.maxx:
            return False
        elif py < self.miny or py >= self.maxy:
            return False
        elif pz < self.minz or pz >= self.maxz:
            return False
        # elif node.x < 0 or node.x>= self.xwidth:
        #     return False
        # elif node.y < 0 or node.y>= self.ywidth:
        #     return False
        # elif node.z < 0 or node.z>= self.zwidth:
        #     return False
            
        # collision check
        if self.obmap[node.x][node.y][node.z]:
            return False
        
        return True
                   
    def get_obstacle_map(self,blocks,boundary):
        
        # real length of obstacles (unit:m)
        self.minx = boundary[0,0]
        self.miny = boundary[0,1]
        self.minz = boundary[0,2]
        self.maxx = boundary[0,3]
        self.maxy = boundary[0,4]
        self.maxz = boundary[0,5]
        print("minx: {a}, maxx {b}".format(a= self.minx, b=self.maxx))
        print("miny: {c}, maxy: {d}".format(c= self.miny, d=self.maxy))
        print("minz: {e}, maxz: {f}".format(e= self.minz, f=self.maxz))
        
        self.xwidth = round((self.maxx - self.minx)/ self.reso).astype(int)
        self.ywidth = round((self.maxy - self.miny)/ self.reso).astype(int)
        self.zwidth = round((self.maxz - self.minz)/ self.reso).astype(int)
        print("xwidth: {a}, ywidth: {b}, zwidth: {c}".format(a=self.xwidth, b=self.ywidth, c=self.zwidth))
        
        # obstacle map generation
        self.obmap = [[[False for i in range(self.zwidth)]for j in range(self.ywidth)]for k in range(self.xwidth)]
        
        # fill the obstacles
        for i in range(blocks.shape[0]): # for each obstacle
            obs = blocks[i]
            # transform the coordinate and regulate
            ox_min, ox_max = self.length2grid(obs[0],self.minx),self.length2grid(obs[3],self.minx)
            oy_min, oy_max = self.length2grid(obs[1],self.miny),self.length2grid(obs[4],self.miny)
            oz_min, oz_max = self.length2grid(obs[2],self.minz),self.length2grid(obs[5], self.minz)
            if ox_max >= self.xwidth:
                ox_max = self.xwidth-1
            # elif oy_max >= self.ywidth:
            #     oy_max = self.ywidth-1
            # elif oz_max >= self.zwidth:
            #     oz_max = self.zwidth-1
            # iterate through each coordinate           
            for bx in range(ox_min, ox_max):
                for by in range(oy_min, oy_max):
                    for bz in range(oz_min, oz_max):
                        #print("x: {a}, y: {b}, z: {c}".format(a=bx,b=by,c=bz))
                        self.obmap[bx][by][bz] = True
    
    @staticmethod
    def expand():
        # dx, dy, dz, cost
        motion = [[1,0,1,math.sqrt(2)] , [0,1,1,math.sqrt(2)]  , [-1,0,1,math.sqrt(2)]  , [0,-1,1,math.sqrt(2)] ,
                  [1,1,1,math.sqrt(3)] , [1,-1,1,math.sqrt(3)] , [-1,-1,1,math.sqrt(3)] , [-1,1,1,math.sqrt(3)] ,
                  [1,0,-1,math.sqrt(2)], [0,1,-1,math.sqrt(2)] , [-1,0,-1,math.sqrt(2)] , [0,-1,-1,math.sqrt(2)],
                  [1,1,-1,math.sqrt(3)], [1,-1,-1,math.sqrt(3)], [-1,-1,-1,math.sqrt(3)], [-1,1,-1,math.sqrt(3)],
                  [1,0,0,1]            , [0,1,0,1]             , [-1,0,0,1]             , [0,-1,0,1]            ,
                  [1,1,0,math.sqrt(2)] , [1,-1,0,math.sqrt(2)] , [-1,-1,0,math.sqrt(2)] , [-1,1,0,math.sqrt(2)] ,
                  [0,0,1,1]            , [0,0,-1,1]]

        return motion
    
    def plot(self, path, start, goal):
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.scatter(path[:,0], path[:,1], path[:,2], c='g');
        ax.scatter(start[0], start[1], start[2], c='r');
        ax.scatter(goal[0], goal[1], goal[2], c='b');
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.show()
    
    
if __name__ == '__main__':
    mapdata = np.loadtxt('./maps/single_cube.txt',dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
    blockIdx = mapdata['type'] == b'block'
    boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
    blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view('<f4').reshape(-1,11)[:,2:]
    myPlanner = Weighted_Astar_Planner(blocks,boundary, eps=1)
    
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 5.5])
    
    path = myPlanner.planning(start, goal)

    myPlanner.plot(path, start, goal)    

    
    

