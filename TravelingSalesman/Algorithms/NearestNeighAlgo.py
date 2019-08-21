import numpy as np
import pandas as pd
import warnings
import math
import time
warnings.filterwarnings('ignore')

class NearestNeighbour:
    def read_data(self, data_file):
        df = pd.read_csv(data_file) 
        nodes = []
        for i in range(len(df['pairs'])):
            sp = df['pairs'][i].split(' ')
            x = float(sp[1])
            y = float(sp[2])
            nodes.append([x, y])
        return nodes
    
    def closestpoint(self, point, route):
        dmin = float("inf")
        for each in route:
            dis = math.sqrt((int(point[0]) - int(each[0]))**2 + (int(point[1]) - int(each[1]))**2)
            if dis < dmin:
                dmin = dis
                closest = each
        return closest, dmin   
    
    

def main(data_file):
    start = time.time()

    nn = NearestNeighbour()

    nodes = nn.read_data(data_file)
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

    print('Shortest Path Length', shortest_distance)

    time_lapse = time.time() - start
    print('Time lapsed', time_lapse)
    
    
if __name__=='__main__':
    main(data_file)