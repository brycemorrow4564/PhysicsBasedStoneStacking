import pymesh 
import os 

base_path = '../write_rocks'
full_rock_mesh_path = f'{base_path}/rocks.obj'
write_single_rocks_path = f'{base_path}/SingleRockFiles_Raw'

full_rock_mesh = pymesh.load_mesh(full_rock_mesh_path)
rock_meshes = pymesh.separate_mesh(full_rock_mesh)

rockIndex = 1 
for rock_mesh in rock_meshes: 
    rock_mesh_path = f"{write_single_rocks_path}/rock{rockIndex}.obj"
    pymesh.save_mesh(rock_mesh_path, rock_mesh)
    print(f"Wrote rock {rockIndex}")
    rockIndex += 1

print("Finished extracting disconnected meshes from a larger mesh defined in a single file!")

