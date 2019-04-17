#!/bin/sh

#This command is to open up a python interpreter inside a docker container with the 
#pymesh library installed. 

docker run -i -t --mount type=bind,src=/Users/ALEX/Documents/RoboticsFinal/Rocks,dst=/write_rocks pymesh/pymesh

#Once we do this, we enter the following command into python 
# >>> import subprocess 
# >>> subprocess.check_output(['python', '../write_rocks/RockMeshToSingleRockMeshes.py])

# The executing script will load the single .obj file which contains all rigid body objects. 
# it will use the functionality of the pymesh library to separate the disconnected components
# into individual triangular mesh objects 