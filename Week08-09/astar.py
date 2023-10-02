import heapq
import math

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current node to goal)
        self.f = 0  # Total cost: f = g + h

def euclidean_distance(node1, node2):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

def a_star(start, goal, obstacles):
    open_list = []
    closed_set = set()
    
    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])
    
    heapq.heappush(open_list, (start_node.f, start_node))
    
    while open_list:
        _, current_node = heapq.heappop(open_list)
        
        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node is not None:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # Reverse the path to get it from start to goal
        
        closed_set.add((current_node.x, current_node.y))
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                neighbor_x = current_node.x + dx
                neighbor_y = current_node.y + dy
                
                if (neighbor_x, neighbor_y) in obstacles or neighbor_x < 0 or neighbor_y < 0:
                    continue
                
                if (neighbor_x, neighbor_y) in closed_set:
                    continue
                
                neighbor_node = Node(neighbor_x, neighbor_y, current_node)
                neighbor_node.g = current_node.g + euclidean_distance(current_node, neighbor_node)
                neighbor_node.h = euclidean_distance(neighbor_node, goal_node)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
    
    return None  # No path found

# Example usage:
start_point = (0, 0)
destination_point = (5, 5)
obstacles = [(1, 1), (2, 2), (3, 3), (4, 4)]

waypoints = a_star(start_point, destination_point, obstacles)
if waypoints:
    print("Waypoints:", waypoints)
else:
    print("No path found.")
