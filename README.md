1. Input is a single .obj file that contains the definitions of any number of disconnected 3-dimensional polyhedra. 
2. I use the pymesh library to read the object defintions from this source file and to generate the corresponding triangular meshes.
3. For each of these triangular meshes, we compute a connected approximated convex decomposition of each of the objects. 
4. This closed convex decomposition forms a convex hull. We take the volume of this convex hull and use it as the mass in our downstream physics simulator, as we assume uniform density of the rigid bodies. 

