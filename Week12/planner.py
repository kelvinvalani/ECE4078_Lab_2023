from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import heapq

class PathPlanner:

    def __init__(self,obstacles,destinations):
        self.obstacles = obstacles
        self.destinations = destinations
        self.grid_size = (300,300)
        self.obstacle_radius = 15
        self.occupancy_grid = []
        self.start = [150, 150]

    def create_occupancy_grid(self):
        # Create an empty grid
        self.occupancy_grid = np.zeros(self.grid_size)

        for obstacle in self.obstacles:
            y, x = obstacle

            # Calculate the range of grid cells affected by the obstacle
            x_range = slice(max(0, x - self.obstacle_radius), min(self.grid_size[0], x + self.obstacle_radius + 1))
            y_range = slice(max(0, y - self.obstacle_radius), min(self.grid_size[1], y + self.obstacle_radius + 1))

            # Mark all cells within the specified radius as occupied
            self.occupancy_grid[x_range, y_range] = 1

        return self.occupancy_grid

    def plot_occupancy_grid(self):
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Occupancy Grid')
        plt.grid()
        plt.show()

    def heuristic(self,node, goal):
        # Calculate the Manhattan distance heuristic
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def astar(self,grid, start, goal):
        start = (start[0], start[1])
        goal = (goal[1], goal[0])
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {pos: float('inf') for pos in np.ndindex(grid.shape)}
        g_score[start] = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    x, y = current[0] + dx, current[1] + dy
                    neighbor_pos = (x, y)

                    if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x, y] == 0:
                        tentative_g_score = g_score[current] + 1

                        if tentative_g_score < g_score[neighbor_pos]:
                            came_from[neighbor_pos] = current
                            g_score[neighbor_pos] = tentative_g_score
                            f_score = tentative_g_score + self.heuristic(neighbor_pos, goal)
                            heapq.heappush(open_list, (f_score, neighbor_pos))

        return None  # No path found

    def plot_path_on_occupancy_grid(self, path_mat):
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.colorbar(label='Occupancy')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Occupancy Grid with Path')
        plt.grid()

        if path_mat:
            path = np.array(path_mat)
            plt.plot(path[:, 1], path[:, 0], marker='o', color='red', markersize=5, label='Path')
            plt.legend()

        plt.show()


    def extract_edges(self,path):
        if not path:
            return []

        edges = [path[0]]
        for i in range(1, len(path) - 1):
            prev_dir = (path[i - 1][0] - path[i][0], path[i - 1][1] - path[i][1])
            curr_dir = (path[i][0] - path[i + 1][0], path[i][1] - path[i + 1][1])
            if prev_dir != curr_dir:
                edges.append(path[i])
        edges.append(path[-1])
        return edges

if __name__ == '__main__':

    # Example usage:
    obstacle_coordinates = [[70, 270], [50, 150], [250, 70], [170, 150], [150, 100], [200, 130], [270, 150], [200, 250], [100, 200], [250, 250], [70, 180], [200, 50], [30, 260], [50, 50], [270, 30], [120, 30], [140, 120], [180, 210], [220, 170]]
    destinations = [[50, 100]]
    planner = PathPlanner(obstacle_coordinates,destinations)
    occupancy_grid = planner.create_occupancy_grid()

    path_mat = []
    waypoints_mat = []
    for i in range(len(destinations)):
        path = planner.astar(occupancy_grid, planner.start, destinations[i])
        waypoints = planner.extract_edges(path)
        path_mat.append(path)
        waypoints_mat.append(waypoints)
        planner.start = destinations[i][::-1]
    if path_mat:
        planner.plot_path_on_occupancy_grid(path_mat[0])
    else:
        print("No path found")
    
