import trimesh 
import shutil
import os 

base_path = '/home/user/write_rocks'
read_single_rocks_path = f'{base_path}/SingleRockFiles_Raw'
write_dir_name = 'SingleRockFiles_Processed'
write_single_rocks_path = f'{base_path}/{write_dir_name}'

#Get the list of all rock files to process 
single_raw_rock_files = [fname for fname in os.listdir(read_single_rocks_path) if '.obj' in fname] 

#Remove all existing entries from the processed rock directory and 
#delete the directory itself 
shutil.rmtree(write_single_rocks_path)

#Recreate the directory, it will be empty now 
os.mkdir(write_single_rocks_path)

for raw_rock_fname in single_raw_rock_files: 
    # Open the file that contains the .obj definition of our rigid bodies
    raw_rock_file = open(f"{read_single_rocks_path}/{raw_rock_fname}")
    # Load this file into a trimesh digestable format
    mesh_kwargs = trimesh.exchange.wavefront.load_wavefront(raw_rock_file)
    # Construct a mesh based on the file definition 
    mesh = trimesh.base.Trimesh(**mesh_kwargs[0])
    # Create a directory where we will write the updated rock files
    rock_urdf_dirname = raw_rock_fname.split(".")[0]
    rock_urdf_dirpath = f'{write_single_rocks_path}/{rock_urdf_dirname}'
    os.mkdir(rock_urdf_dirpath)
    # The following line does multiple things 
    # 1. Performs a convex decomposition (using VHACD algorithm) 
    #    to split the raw mesh into a small number of convex hulls 
    # 2. Writes multiple .obj files (one for each convex piece) and a single
    #    .urdf file which we will use as the definition when loading the 
    #    object into pyBullet.  
    trimesh.exchange.urdf.export_urdf(mesh, rock_urdf_dirpath)

print('''
Finished 
1. VHACD
2. .obj -> .urdf conversion!
''')

