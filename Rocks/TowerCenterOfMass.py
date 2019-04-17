import scipy
import numpy as np 


def geometric_centroid_of_convex_hull(hull):
    #hull.points is all of the points that are contained in the hull
    #hull.vertices is an index into points giving all points that lie on the hull
    #np.mean with axis = 0 takes columnwise sum (3 cols for 3D space)
    geom_centroid = np.mean(hull.points[hull.vertices], axis=0)
    return geom_centroid

def convex_decomposition_center_of_mass_calculation(convex_hulls): 
    '''
    Compute the center of mass over a collection (list) of convex hulls
    Every input convex hull is of the type scipy.spatial.ConvexHull
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    '''
    #centroids for each individual convex hull
    centroids = [geometric_centroid_of_convex_hull(hull) for hull in convex_hulls]
    #mass for each individual convex hull. use volume as analog as we assume uniform density 
    masses = [hull.volume for hull in convex_hulls]
    #mass over all hulls, which we use to normalize weights
    total_mass = np.sum(masses)
    #weights determine how much each centroid contributes to the overall center of mass 
    weights = masses / total_mass

    #form a convex combination of weights to compute the final centroid 
    centroid_total = [0,0,0]
    for ctroid,weight in zip(centroids, weights): 
        centroid_total[0] += ctroid[0] * weight
        centroid_total[1] += ctroid[1] * weight 
        centroid_total[2] += ctroid[2] * weight 

    #The final centroid is the center of mass of the collection of convex hulls
    return centroid_total


