"""
Code Reference : https://github.com/DiegoVicen/som-tsp   
"""
import numpy as np
import pandas as pd
import time
import math

class SelfOrganizingMap:
    
    #Return the normalized version of a given vector of points
    def normalize(self, points):
        ratio = (points['x'].max() - points['x'].min()) / (points['y'].max() - points['y'].min()), 1
        ratio = np.array(ratio) / max(ratio)
        norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
        return norm.apply(lambda p: ratio * p, axis=1)
    
    
    def between_distance(self, lat1, long1, lat2, long2):
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
    
    
    #def select_closest(self, candidates, origin):
        #Return the index of the closest candidate to a given point
    #    return self.eucledian_distance(candidates, origin).argmin()
    
    def select_closest(self, route, point):
        try:
            dmin = float("inf")
            for i in range(len(route)):
                each = route[i]
                #dis = math.sqrt((int(point[0]) - int(each[0]))**2 + (int(point[1]) - int(each[1]))**2)
                dis = self.between_distance(point[0], point[1], each[0], each[1])
                if dis < dmin:
                    dmin = dis
                    closest = i
            return closest
        except:
            dmin = float("inf")
            for i in range(len(route)):
                each = route[i]
                #dis = math.sqrt((int(point[0]) - int(each[0]))**2 + (int(point[1]) - int(each[1]))**2)
                dis = self.between_distance(point[0][0], point[0][1], each[0], each[1])
                if dis < dmin:
                    dmin = dis
                    closest = i
            return closest
    

    def eucledian_distance(self, a, b):
        return np.linalg.norm(a - b, axis = 1)
    
    
    def route_distance(self, cities):
        #Return the cost of traversing a route of cities in a certain order
        points = cities[['x', 'y']]
        #distances = self.eucledian_distance(points, np.roll(points, 1, axis=0))
        distances = []
        for i in range(len(points)-1):
            distances.append(self.between_distance(points['x'][i], points['y'][i], points['x'][i+1], points['y'][i+1]))
        return np.sum(distances)
    
    
    
    def generate_network(self, size):
        #Generate neuron network for a given size
        #Return a vector of 2-D points in the interval of [0,1]
        return np.random.rand(size, 2)

    
    def get_neighborhood(self, center, radix, domain):
        #Get the range gaussian of given radix around a center index
        #print(center, radix, domain)
        if radix < 1:
            radix = 1

        deltas = np.absolute(center - np.arange(domain))
        distances = np.minimum(deltas, domain - deltas)

        return np.exp(-(distances*distances) / (2*(radix*radix)))

    
    def get_route(self, cities, network):
        #Return the route computed by a network
        cities['winner'] = cities[['x', 'y']].apply(
        lambda c: self.select_closest(network, c), 
        axis=1, raw=True)

        return cities.sort_values('winner').index
    
        
        
    def som(self, problem, iterations, learning_rate=0.8):
        """Solve the TSP using a Self-Organizing Map."""

        # Obtain the normalized set of cities (w/ coord in [0,1])
        cities = problem.copy()

        cities[['x', 'y']] = self.normalize(cities[['x', 'y']])

        # The population size is 8 times the number of cities
        n = cities.shape[0] * 8

        # Generate an adequate network of neurons:
        network = self.generate_network(n)
        #print('Network of {} neurons created. Starting the iterations:'.format(n))

        for i in range(iterations):
            #if not i % 100:
                #print('\t> Iteration {}/{}'.format(i, iterations), end="\r")
            # Choose a random city
            city = cities.sample(1)[['x', 'y']].values
            winner_idx = self.select_closest(network, city)
            # Generate a filter that applies changes to the winner's gaussian
            gaussian = self.get_neighborhood(winner_idx, n//10, network.shape[0])
            # Update the network's weights (closer to the city)
            network += gaussian[:,np.newaxis] * learning_rate * (city - network)
            # Decay the variables
            learning_rate = learning_rate * 0.99997
            n = n * 0.9997
            
        route = self.get_route(cities, network)
        return route

    
def main(nodes):
    maps = SelfOrganizingMap()
    nodes = pd.DataFrame(nodes, columns=['x', 'y'])
    route = maps.som(nodes, 100000)
    nodes1 = nodes.reindex(route)
    distance = maps.route_distance(nodes1)
    return nodes1.values.tolist(), distance


if __name__=='__main__':
    main(nodes)