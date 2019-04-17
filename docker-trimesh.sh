#This command is to open up a python interpreter inside a docker container with the 
#trimesh library installed. 

docker run -it --mount type=bind,src=/Users/ALEX/Documents/RoboticsFinal/Rocks,dst=/home/user/write_rocks mikedh/trimesh

#Once we do this, we enter the following command into python 
# >>> import subprocess 
# >>> subprocess.check_output(['python', '../write_rocks/ConvexDecompositionVHACD.py])

# The executing script will load all individual rock .obj files and apply VHACD, which decomposes
# a triangular mesh object into a set of convex hulls. This allows us to more simply model collision 
# checking in pyBullet 
