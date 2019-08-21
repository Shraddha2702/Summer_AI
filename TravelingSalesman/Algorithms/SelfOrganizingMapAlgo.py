import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

class SelfOrganizingMap: 
    def read_data(self, data_file):
        df = pd.read_csv(data_file) 
        nodes = []
        for i in range(len(df['pairs'])):
            sp = df['pairs'][i].split(' ')
            x = float(sp[1])
            y = float(sp[2])
            nodes.append([x, y])
        return pd.DataFrame(nodes, columns=['x', 'y'])
    
    #Return the normalized version of a given vector of points
    def normalize(self, points):
        ratio = (points['x'].max() - points['x'].min()) / (points['y'].max() - points['y'].min()), 1
        ratio = np.array(ratio) / max(ratio)
        norm = points.apply(lambda c: (c - c.min()) / (c.max() - c.min()))
        return norm.apply(lambda p: ratio * p, axis=1)
    
    
    
    def select_closest(self, candidates, origin):
        #Return the index of the closest candidate to a given point
        return self.eucledian_distance(candidates, origin).argmin()

    def eucledian_distance(self, a, b):
        return np.linalg.norm(a - b, axis = 1)
    
    def route_distance(self, cities):
        #Return the cost of traversing a route of cities in a certain order
        points = cities[['x', 'y']]
        distances = self.eucledian_distance(points, np.roll(points, 1, axis=0))
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

    
    def plot_route(self, cities, route, name='diagram.png', ax=None):
        """Plot a graphical representation of the route obtained"""
        mpl.rcParams['agg.path.chunksize'] = 10000

        if not ax:
            fig = plt.figure(figsize=(5, 5), frameon = False)
            axis = fig.add_axes([0,0,1,1])

            axis.set_aspect('equal', adjustable='datalim')
            plt.axis('off')

            axis.scatter(cities['x'], cities['y'], color='red', s=4)
            route = cities.reindex(route)
            route.loc[route.shape[0]] = route.iloc[0]
            axis.plot(route['x'], route['y'], color='purple', linewidth=1)

            plt.savefig(name, bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close()

        else:
            ax.scatter(cities['x'], cities['y'], color='red', s=4)
            route = cities.reindex(route)
            route.loc[route.shape[0]] = route.iloc[0]
            ax.plot(route['x'], route['y'], color='purple', linewidth=1)
            return ax
        
        
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

            # Check if any parameter has completely decayed.
            #if n < 1:
                #print('Radius has completely decayed, finishing execution',
                #'at {} iterations'.format(i))
            #    break
            #if learning_rate < 0.001:
                #print('Learning rate has completely decayed, finishing execution',
                #'at {} iterations'.format(i))
            #    break
        #else:
            #print('Completed {} iterations.'.format(iterations))

        route = self.get_route(cities, network)
        #self.plot_route(cities, route, 'route.png')
        return route

    
def main(data_file):
    start = time.time()
    
    maps = SelfOrganizingMap()
    nodes = maps.read_data(data_file)
    route = maps.som(nodes, 100000)
    nodes1 = nodes.reindex(route)
    distance = maps.route_distance(nodes1)

    print('Shortest Path Length', distance)
    time_lapse = time.time() - start
    print('Time lapsed', time_lapse)


if __name__=='__main__':
    main(data_file)