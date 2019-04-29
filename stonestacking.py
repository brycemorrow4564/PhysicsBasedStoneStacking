
# coding: utf-8

# # Sequential Rigid Body Object Stacking
# 
# A greedy, physics-based next best object pose planning algorithm 

# In[1]:


import pybullet as p
import os 
import time
import pybullet_data
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns

from Rocks.TowerCenterOfMass import convex_decomposition_center_of_mass_calculation

get_ipython().run_line_magic('matplotlib', 'inline')

'''
Utility functions
'''
lmap = lambda fn,it: list(map(fn,it))

def in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

'''
File System Settings
'''
FINAL_DATA_PATH = './Rocks/SimulatorRockObjDefinitions'
DATA_DIR = './Rocks/SingleRockFiles_Processed'
ROCK_DIRS = os.listdir(DATA_DIR)

'''
Physics Engine Settings 
'''
MESH_SCALE = .03
KINETIC_ENERGY_STABLE = 50
CONTACT_ITERS = 50
GRAVITY = -10


# In[2]:


class RandomizedPoseSampler(): 
    
    def __init__(self, testloc=[-10000,-10000,-10000]): 
        '''
        The RandomizedPoseSampler exposes a single function interface. 
        
        For a given rock, the randomized pose sampler 
        1. generates a random quaternion.
        2. shift object to arbitrary location at given orientation
        3. compute aabb. 
        4. use aabb to determine offset from bbox base to object com 
           in the z dimension. 
        5. translate the rock to be directly above the top of the tower 
           (but not intersecting the top plane of the tower) and have the
           object's COM be vertically aligned with the geometric centroid
           of the polygon of support of the tower. 
        '''
        self.testloc = testloc

    def sample_and_execute_pose(self, rock, tower_com, tower_top_z):
        
        rock.set_gravity(False)

        #Generate a randomized quaternion 
        rand_quat = p.getQuaternionFromEuler([np.random.uniform(0, np.pi*2) for i in range(3)])
        
        # Move the rock center of mass to the testing location. At this location, we perform the 
        # rotation specified by our randomly sampled quaternion. 
        rock.set_pos_ori(self.testloc, rand_quat)
        
        # Compute the bounding box of the object positioned at the test location
        new_bbox = p.getAABB(rock.sim_id)
        
        # Compute the signed vertical (z) distance between the 
        # tower top and bottom of the new bbox 
        dz = self.testloc[2] - new_bbox[0][2]
        epsilon = .01
        dz += epsilon
        
        new_pos = [tower_com[0], tower_com[1], tower_top_z + dz]
        
        '''
        The random position that results from placing an object, at the specified orientation 
        such that the lowest point in it's aabb is dz units above the maximum z value 
        of the aabb of the tower. 
        '''
        rock.set_pos_ori(new_pos, rand_quat)
        
        assert(p.getAABB(rock.sim_id)[0][2] < tower_top_z, 'randomly sampled pose intersects top of tower plane')
                
        # Return the new pose of the rock  
        return new_pos, rand_quat


# In[3]:


class NextBestPoseSearch: 
    
    def __init__(self, rock_tower, non_tower_rocks, n_iters_per_rock=3): 
        
        # The tower that contains all already stacked rocks 
        self.rock_tower = rock_tower
        
        # Rocks that we can consider candiates to place on top 
        # of the existing tower during the pose planning phase 
        self.non_tower_rocks = non_tower_rocks
        
        # The number maximum number of attempts we make 
        # to find a valid pose for a given rock 
        self.n_iters_per_rock = n_iters_per_rock
        
        
    def pose_cost(self, rock): 
        
        # Checkpoint tower
        self.rock_tower.hash_state()
        
        # Compute the cost of the pose 
        ke = rock.kinetic_energy()
        
        # Compute the starting position of the rock 
        ret_pos, ret_ori = rock.get_pos_ori()
        
        # Enable gravity for rock and tower 
        self.rock_tower.set_gravity(True)
        rock.set_gravity(True)
        
        # Iterate, stepping simulation until we have contacts 
        contacts = None 
        for i in range(CONTACT_ITERS): 
            p.stepSimulation()
            contacts = p.getContactPoints(rock.sim_id)
            
        cost = None
            
        # Infinite cost if object does not settle into contact position
        if len(contacts) < 3:
            cost = np.inf
        else: 
            # compute newly formed polygon of support between the new and 
            # old tower top objects 
            points = lmap(lambda c: [c[5][0], c[5][1]], contacts)
            try: 
                # Cost is low for high hull area and vice versa 
                hull = ConvexHull(points)
                area = hull.volume 
                cost = 1./area
            except Exception as e: 
                # Infinite cost if the hull is not convex 
#                 print('hull failed: ', points)
                cost = np.inf

        # Disable gravity for rock and tower 
        self.rock_tower.set_gravity(True)
        rock.set_gravity(True)
        
        # Revert the rock and rock tower to the checkpointed poses 
        rock.set_pos_ori(ret_pos, ret_ori)
        self.rock_tower.revert_to_hash_state()
        
        return cost
            
    def run(self): 
        
        canStack = True 
        posesampler = RandomizedPoseSampler()
        
        #freeze all non-tower rocks
        lmap(lambda r: r.set_gravity(False), self.non_tower_rocks)
        
        height = None
                
        while True: 
            
            # indicator variable of any successes
            canStack = False
            
            # reset each iteration 
            least_cost = np.inf
            best_pose = None
            best_rock = None
            
            # compute the current tower center of mass
            # approximated as weighted geometric centroid of tower 
            tower_com = self.rock_tower.get_com()
            
            # compute the current tower top plane (defined by single z value)
            tower_top_z = self.rock_tower.get_top_plane()
            
            # non tower rocks
            non_tower_rocks = self.non_tower_rocks
            
#             print(f"tower top z: {tower_top_z}")
            
            # randomly sample poses to find low cost valid poses 
            for rock in non_tower_rocks: 
                
                ret_pos, ret_ori = rock.get_pos_ori()
                
                for i in range(self.n_iters_per_rock): 
                
                    rand_pos, rand_ori = posesampler.sample_and_execute_pose(rock, tower_com, tower_top_z)
                    
                    # Rock is now hovering above the tower at a specified position and orientation 
                    # we need to step the simulation a few times with gravity enabled until a 
                    # collision occurs 
                    is_valid_pose, kinetic_energy = self.rock_tower.valid_pose_test(rock)
                    
                    if is_valid_pose: 
                        
                        # Compute the cost of the current pose of the rock 
                        # pose was frozen after completion of valid_pose_test
                        pos, ori = rock.get_pos_ori()
                        
                        #Compute the cost 
                        cost = self.pose_cost(rock)
                        
                        if cost < least_cost: 
                            canStack = True 
                            least_cost = cost 
                            best_pose = (pos, ori)
                            best_rock = rock
#                             print(f'NEW LOWEST COST POSE: {best_rock.sim_id} - {least_cost}')
                        else: 
#                             print(f'non-lowest cost pose: {rock.sim_id} - {cost}')
                            pass 
                    else: 
#                         print(f'unstable kinetic energy: {rock.sim_id} - {kinetic_energy}')
                        pass

                rock.set_gravity(False)
                rock.set_pos_ori(ret_pos, ret_ori)     
        
                cur_pos, cur_ori = rock.get_pos_ori()
            
                # Ensure the start and end states of the rock are equivalent 
                np.testing.assert_array_almost_equal(cur_pos, ret_pos)
                np.testing.assert_array_almost_equal(cur_ori, ret_ori)
                
            # If we found a rock to stack, we stack it 
            # If we did not, the loop will terminate when the while condition is not met 
            if canStack: 
#                 print(f'stacking: {best_rock.sim_id}')
                best_rock.set_pos_ori(best_pose[0], best_pose[1])
                self.rock_tower.add_rock(best_rock)
                prelen = len(self.non_tower_rocks)
                self.non_tower_rocks = [r for r in self.non_tower_rocks if r.sim_id != best_rock.sim_id]
                postlen = len(self.non_tower_rocks)
                assert(prelen == postlen + 1)
            else: 
#                 print(f'\nFINISHED STACKING - Height: {self.rock_tower.height()} objects', '\n')
                height = self.rock_tower.height()
                break 
                
        return height
                


# In[4]:


class Rock: 
    '''
    Abstraction representing our physics simulator rock object
    '''
    
    def __init__(self, rock_path, hulls, com, volume): 
        self.rock_path = rock_path
        self.hulls = hulls 
        self.com = com
        self.com_inv = lmap(lambda e: -e, com) 
        self.mass = volume
        self.gravityEnabled = True 
        self.initialPosition = [0,0,0]
        self.initialPosition[0] += self.com_inv[0]
        self.initialPosition[1] += self.com_inv[1]
        self.initialPosition[2] += self.com_inv[2]
        self.sim_id = None #will remain None until rock is spawned in simulation 
        
    def set_gravity(self, gravityOn): 
        assert(self.sim_id)
        self.gravityEnabled = gravityOn
        if self.gravityEnabled: 
            p.changeDynamics(self.sim_id, -1, mass=self.mass)
        else: 
            p.changeDynamics(self.sim_id, -1, mass=0)
        
    def get_gravity_enabled(self): 
        assert(self.sim_id)
        return self.gravityEnabled
        
    def get_sim_id(self):
        assert(self.sim_id)
        return self.sim_id
    
    def get_mass(self): 
        return self.mass 
    
    def get_com(self): 
        # The position of the base is the object center of mass 
        assert(self.sim_id)
        cur_pos, cur_ori = p.getBasePositionAndOrientation(self.sim_id)
        return cur_pos 
        
    def get_pos_ori(self):
        assert(self.sim_id)
        return p.getBasePositionAndOrientation(self.sim_id)
    
    def kinetic_energy(self): 
        '''
        https://www.real-world-physics-problems.com/kinetic-energy.html
        Formula derivation for kinetic energy of 3D rigid body 
        '''
        m = self.mass
        v_linear, v_angular = p.getBaseVelocity(self.sim_id)
        v_lin_norm = np.linalg.norm(v_linear)
        wx,wy,wz = v_angular
        inertia = p.getDynamicsInfo(self.sim_id, -1)[2]
        ix,iy,iz = inertia
        return ((.5*m*v_lin_norm**2) + 
                (.5*ix*wx**2) + 
                (.5*iy*wy**2) + 
                (.5*iz*wz**2))
    
    def set_pos_ori(self, pos=None, ori=None):
        '''
        Set the position and or orientation of the rock in the simulation 
        '''
        assert(self.sim_id)
        cur_pos, cur_ori = p.getBasePositionAndOrientation(self.sim_id)
        do_change = pos is not None or ori is not None 
        
        # is position or orientation is unspecified, simply keep previous values 
        if pos is None: 
            pos = cur_pos
        if ori is None: 
            ori = cur_ori
            
        # only re-render if necessary 
        if do_change: 
            p.resetBasePositionAndOrientation(self.sim_id, pos, ori)
    
    def spawn(self, hasGravity=True): 
        '''
        Builds a rock from a .obj definition file (visual / collision components)
        Spawns the rock at a given location in the physics simulator (relative to center of mass)

        Returns: 
            String: The id of the created object 
        '''
        
        mass = self.mass if hasGravity else 0 

        #Create visual component 
        visualId = p.createVisualShape(p.GEOM_MESH, 
                                       fileName=self.rock_path, 
                                       meshScale=[MESH_SCALE for i in range(3)])
        #Create colliosion component 
        collisionId = p.createCollisionShape(p.GEOM_MESH, 
                                             fileName=self.rock_path, 
                                             meshScale=[MESH_SCALE for i in range(3)])
        
        #Link visual/collison components to create dynamic rigid body in environment 
        rockId = p.createMultiBody(baseCollisionShapeIndex=collisionId, 
                                   baseVisualShapeIndex=visualId, 
                                   baseMass=mass, 
                                   basePosition=self.initialPosition, 
                                   baseInertialFramePosition=self.com)
        
        #add dynamic properties to the rigid body 
        p.changeDynamics(rockId, 
                         -1, #every multi-body consists of only a single base link, referenced as -1
                         lateralFriction=.4, #contact friction
                         restitution=.1) #bouncyness 
        
        #update sim_id from None to the spawned object id 
        self.sim_id = rockId
        


# In[ ]:


class RockTower: 
    '''
    Abstraction to encapsulate information about the tower of rocks. 
    
    [Rock] - rocks: 
        A list of rock objects 
    '''
    
    def __init__(self, defaultPosition, planeId):
        self.planeId = planeId
        self.defaultPosition = defaultPosition
        #Tower always starts empty 
        self.rocks = []
        self.posehash = {}
        
    def height(self): 
        return len(self.rocks)
        
    def get_com(self): 
        '''
        Get the center of mass of the tower 
        '''
        if len(self.rocks) == 0:
            return self.defaultPosition
        coms = [rock.get_com() for rock in self.rocks]
        masses = [rock.get_mass() for rock in self.rocks]
        total_mass = sum(masses)
        weighted_coms = []
        for com, mass in zip(coms, masses):
            mass_p = mass / total_mass
            weighted_coms.append(lmap(lambda coord: coord * mass_p, com))
        weighted_coms = np.array(weighted_coms).reshape((-1,3))
        return np.mean(weighted_coms, axis=0)
    
    def get_top_plane(self, epsilon=.001):
        '''
        Return the z value for a plane that lies epsilon units 
        directly above the top of the bounding box of the current
        top of the tower. If there are no rocks in the tower yet, 
        we return a plane right above the default position 
        '''
        if len(self.rocks) == 0: 
            return self.defaultPosition[2] + epsilon
        else: 
            toprock = self.rocks[-1]
            toprock_bbox = p.getAABB(toprock.sim_id)
            return toprock_bbox[1][2] + epsilon
        
    def valid_pose_test(self, arock): 

        # Enable gravity for the object being placed on top of the tower 
        arock.set_gravity(True)
        # Disable gravity for all objects in the tower 
        self.set_gravity(False)
        
        contacts = []
        contactWithId = self.planeId if len(self.rocks) == 0 else self.rocks[-1].sim_id
        base_hull = self.get_base_polygon_of_support()
        
#         for i in range(1, len(base_hull.points)):
#             p1 = list(base_hull.points[i-1]) + [0]
#             p2 = list(base_hull.points[i]) + [0]
#             p.addUserDebugLine(p1, p2, [1,0,0])
#         p.addUserDebugLine(list(base_hull.points[0]) + [0], list(base_hull.points[-1]) + [0], [1,0,0])
#         for po in base_hull.points: 
#             po = list(po)
#             p.addUserDebugLine(po + [-10], po + [0], [0,1,0])
            
        # Run simulator until object comes to rest 
        # or it falls off the top of the stack 
        contact_iters_max = 1250
        contact_iters = 0 
        
        while contact_iters < contact_iters_max: 
                        
                # Apply a downward force and step the simulation forward 
                p.setGravity(0, 0, GRAVITY)
                p.stepSimulation()
                
                # Compute new COM of object
                com = arock.get_com()
                
                # Determine whether or not COM of stacked object in polygon of support
                com_x, com_y, _ = com
                com_xy = [com_x, com_y]
                if not in_hull(com_xy, base_hull): 
                    return False, np.inf
                
                # Compute the new contacts 
                contacts = p.getContactPoints(arock.sim_id, contactWithId)
                
                # Keep a running count of the number of iterations in a row where the 
                # set of contacts points indicates a stable collision state for the 
                # object and the top of the tower 
                if len(contacts) >= 3: 
                    contact_iters += 1
                else: 
                    contact_iters = 0
                
                if contact_iters == contact_iters_max: 
                    # If we have been in contacts for a number of iterations in a row
                    # we assume that the object has stopped moving 
                    break
        
        kinetic_energy = arock.kinetic_energy()
        
        arock.set_gravity(False)
                
        # If false, then this is an invalid pose. If true, this is a valid pose
        # we return the contacts and kinetic energy for use in cost function 
        return kinetic_energy < KINETIC_ENERGY_STABLE, kinetic_energy
        
    def get_base_polygon_of_support(self): 
        '''
        returns a scipy.spatial.ConvexHull for easy intersection testing 
        '''
        points = None 
        
        if len(self.rocks) == 0: 
            # If there is no base of the tower, return a square centered 
            # around the default positon
            pos = self.defaultPosition
            dv = 100000
            points = [[pos[0]-dv, pos[1]-dv],
                      [pos[0]-dv, pos[1]+dv],
                      [pos[0]+dv, pos[1]+dv],
                      [pos[0]+dv, pos[1]-dv]]
        else: 
            # get contacts between plane and first placed object 
            contacts = p.getContactPoints(self.planeId)
            self.hash_state()
            self.set_gravity(True)
            for i in range(CONTACT_ITERS): 
                p.setGravity(0, 0, GRAVITY)
                p.stepSimulation()
                contacts = p.getContactPoints(self.planeId)
            self.set_gravity(False)
            self.revert_to_hash_state()
            p.stepSimulation()
            points = lmap(lambda c: c[5], contacts)
            points = lmap(lambda co: [co[0], co[1]], points)
                
        assert (points is not None)
        
        try: 
            return ConvexHull(points, qhull_options='Pp')
        except Exception as e: 
            # Rare case where hull has such small angles that 
            # a precision error is encountered, return a default hull
            # that has no chance of containing any points in the stack
            return ConvexHull([[0,.01],
                               [.01,0],
                               [0,0]])
            
    def set_gravity(self, gravityEnabled): 
        '''
        Set the gravity to on / off for all objects in the tower 
        '''
        for rock in self.rocks: 
            rock.set_gravity(gravityEnabled)
        
    def revert_to_hash_state(self): 
        '''
        Revert to the last stable tower state where all object have gravity disabled 
        and exist at a fixed position and orientation 
        '''
        for rock in self.rocks:
            pose = self.posehash[rock.sim_id]
            rock.set_gravity(False)
            rock.set_pos_ori(pos=pose['pos'], 
                             ori=pose['ori'])
        
    def hash_state(self): 
        '''
        Prior to re-enabling gravity, we call this method to record the stable state 
        values for each rock in the tower. If when computing the cost, these values change at 
        all, we reset the tower before the next cost computation 
        '''
        for rock in self.rocks: 
            pos, ori = p.getBasePositionAndOrientation(rock.sim_id)
            self.posehash[rock.sim_id] = { "pos": pos, "ori": ori }
            
    def add_rock(self, rock): 
        self.rocks.append(rock)
        
        


# In[ ]:


def construct_rocks(logging=True):
    '''
    Iterate over a collection of rock files. Each file needs to be parsed 
    to compute center of mass and volume approximations which are stored
    for later use during the physics engine setup / stacking algorithm 
    '''
    rocks = []

    for rdir in ROCK_DIRS: 
        
        obj_files = [f for f in os.listdir(f"{DATA_DIR}/{rdir}") if '.obj' in f]
        obj_paths = [f'{DATA_DIR}/{rdir}/{f}' for f in obj_files]

        # Ensure at least one object file exists 
        assert(len(obj_files) > 0)

        o_counter = 0
        new_file = ''
        hulls = []
        total_volume = 0
        for objp in obj_paths: 
            new_file += f'o object{o_counter}\n'
            o_counter += 1
            with open(objp) as objfile: 
                file_part = objfile.read()
                file_lines = [fl for fl in file_part.split('\n') if fl[0] == 'v'] 
                hull_coords = [lmap(lambda a : MESH_SCALE * float(a), fl.split(' ')[1:])
                               for fl in file_lines]
                hull = ConvexHull(hull_coords)
                total_volume += hull.volume 
                hulls.append(hull)
                new_file += file_part + '\n\n'
        final_rock_path = f'{FINAL_DATA_PATH}/{rdir}.obj'
        com = convex_decomposition_center_of_mass_calculation(hulls)

        rock = Rock(final_rock_path, hulls, com, total_volume)
        rocks.append(rock)

        with open(final_rock_path, 'w') as write_rock_file: 
            write_rock_file.write(new_file)

        if logging: 
            print(f'wrote: {final_rock_path}')
    
    return rocks


# In[ ]:


def spawn_operable_rocks(rocks, initialPosition=[-20,30,0]): 
    '''
    Spawn a collection of static rock objects over which our physics based algorithm will
    operate. Ensure that none of the rocks intersect with each other and that none of the 
    rocks are near the stacking tower. These spawned rocks will have NO gravity. 
    '''
    e = .05
    last_bbox = [[initialPosition[0] - e, initialPosition[1] - e, initialPosition[2] - e],
                 [initialPosition[0] + e, initialPosition[1] + e, initialPosition[2] + e]]
        
    for rock in rocks:  
                
        # Spawn the rock in the simulation, compuate the initial axis aligned bounding box 
        rock.spawn(hasGravity=True)
        new_bbox = p.getAABB(rock.sim_id)
        
        # Compute a transformation that places the new bounding box the the right 
        # of the existing bounding box by computing a change of basis transformation 
        # between the 3d vectors to_pt and from_pt 
        to_pt =   [last_bbox[1][0],
                  (last_bbox[0][1] + last_bbox[1][1]) / 2,
                  (last_bbox[0][2] + last_bbox[1][2]) / 2]
        
        from_pt = [new_bbox[0][0],
                  (new_bbox[0][1] + new_bbox[1][1]) / 2,
                  (new_bbox[0][2] + new_bbox[1][2]) / 2]
        
        # Compute the coordinate shift that must be applied to the object 
        dx = to_pt[0] - from_pt[0]
        dy = to_pt[1] - from_pt[1]
        dz = to_pt[2] - from_pt[2]
        
        cur_pos, cur_ori = rock.get_pos_ori()
        
        new_pos = [
            cur_pos[0] + dx,
            cur_pos[1] + dy,
            cur_pos[2] + dz
        ]
        
        # Set the new rock position 
        rock.set_pos_ori(pos=new_pos)
        
        # update the old bbox 
        last_bbox = p.getAABB(rock.sim_id)
        


# In[ ]:


def axis_aligned_rays(vec3, dv): 
    return [
        [vec3[0] + dv, vec3[1], vec3[2]],
        [vec3[0], vec3[1] + dv, vec3[2]],
        [vec3[0], vec3[1], vec3[2] + dv]
    ]

def add_axis_aligned_rays_at_point(vec3, dv=5):
    lines = axis_aligned_rays(vec3)
    p.addUserDebugLine(vec3, lines[0], [1,0,0])
    p.addUserDebugLine(vec3, lines[1], [0,1,0])
    p.addUserDebugLine(vec3, lines[2], [0,0,1])
    


# In[ ]:


# Enable / disable openGL renderer 
USE_GUI = False 
CONNECTION_TYPE = p.GUI if USE_GUI else p.DIRECT 

# Setup the physics engine. 
physicsClient = p.connect(CONNECTION_TYPE) #or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0) #world has no forces to start


# In[ ]:


# Algorithm Parameters
planePosition = [0,0,-10]
parameter_grid = [
    {
        'n_iters_per_rock': 5,
        'use_n_rocks': 10
    },
    {
        'n_iters_per_rock': 10,
        'use_n_rocks': 10
    },
    {
        'n_iters_per_rock': 20,
        'use_n_rocks': 10
    },
    {
        'n_iters_per_rock': 30,
        'use_n_rocks': 10
    },
]
num_iters_for_params = 10
num_rocks = len([f for f in os.listdir(FINAL_DATA_PATH) if '.obj' in f])
np.random.seed(38)
    
def run_pose_search(rock_index, n_iters_per_rock): 

    # Create a platform upon which we will perform stacking 
    planeId = p.loadURDF("plane.urdf", basePosition=planePosition)

    # Construct and spawn rocks
    rocks = construct_rocks(logging=False)
    spawn_rocks = np.array(rocks)[rock_index]
    spawn_operable_rocks(spawn_rocks)
    
    # Create a rock tower object 
    rock_tower = RockTower(planePosition, planeId)

    # Initiate the next best pose search, returns height of formed tower 
    height = NextBestPoseSearch(rock_tower, spawn_rocks, n_iters_per_rock=n_iters_per_rock).run()
    
    return height 
    
def reset_scene(): 
    p.resetSimulation()

def run_experiment(parameter_grid):
    
    results = []
    
    for paramset in parameter_grid: 

        # Unpack the current parameter values 
        n_iters_per_rock = paramset['n_iters_per_rock']
        use_n_rocks = paramset['use_n_rocks']
        heights = []

        print(f'n_iters_per_rock: {n_iters_per_rock}')
        print(f'use_n_rocks: {use_n_rocks}')

        for i in range(num_iters_for_params): 

            # Reset the existing simulation 
            reset_scene()

            # The index into the collection of all object file paths. 
            # This allows us to select a random subset of all objects without replacement
            # for testing in the physics environment 
            rock_index = np.random.choice(num_rocks, size=use_n_rocks, replace=False)

            # Re-initialize the scene 
            height = run_pose_search(rock_index, n_iters_per_rock)
            heights.append(height)
        
        print(heights,'\n')
        
        results.append([n_iters_per_rock, 
                        use_n_rocks, 
                        heights])
    
    return results 
    
results = run_experiment(parameter_grid)

print(results)


# In[ ]:


legend_elems = []

for i,res in enumerate(results): 
    n_iters_per_rock, use_n_rocks, heights = res
    legend_elems.append(f'n_iters_per_rock: {n_iters_per_rock} - n_rocks: {use_n_rocks}')
    sns.distplot(heights, hist=False, kde=True, ax=plt.gca())

plt.gca().legend(legend_elems)
plt.gcf().set_size_inches((12,8))

