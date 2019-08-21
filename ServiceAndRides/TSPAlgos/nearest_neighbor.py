import numpy as np
import pandas as pd
import math
import time
import math 

class NearestNeighbour:
    def distance(self, lat1, long1, lat2, long2):
        """
        start_x, start_y, end
        """
        degrees_to_radians = math.pi/180.0
        phi1 = (90.0 - lat1)*degrees_to_radians
        phi2 = (90.0 - lat2)*degrees_to_radians

        theta1 = long1*degrees_to_radians
        theta2 = long2*degrees_to_radians

        a = ((math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2)) +(math.cos(phi1)*math.cos(phi2)))
        if a>1:
            a=0.999999
        dis = math.acos( a )
        return dis*6373

    
    def closestpoint(self, point, route):
        dmin = float("inf")
        for each in route:
            #dis = math.sqrt((int(point[0]) - int(each[0]))**2 + (int(point[1]) - int(each[1]))**2)
            dis = self.distance(point[0], point[1], each[0], each[1])
            if dis < dmin:
                dmin = dis
                closest = each
        return closest, dmin   
    



def main(nodes):
    nn = NearestNeighbour()
    
    point, *route = nodes
    path = [point]

    shortest_distance = 0
    while len(route) >= 1:
        closest, dist = nn.closestpoint(path[-1], route)
        path.append(closest)
        route.remove(closest)
        shortest_distance += dist

    # Go back the the beginning when done.
    closest, dist = nn.closestpoint(path[-1], [point])
    path.append(closest)
    shortest_distance += dist
    return path, shortest_distance
    
    
if __name__=='__main__':
    main(nodes)