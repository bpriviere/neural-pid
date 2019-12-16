"""
Script to create random MAPF instance
"""

import os
import collections
import random
import copy
import yaml


def reachable(map_size, start, goal, obstacles):
    visited = set()
    stack = [tuple(start)]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            if vertex == tuple(goal):
                return True
            visited.add(vertex)
            for delta in [[1,0], [-1,0], [0,1], [0,-1]]:
                pos = (vertex[0] + delta[0],vertex[1] + delta[1])
                if pos[0] >= 0 and pos[0] < map_size[0] and pos[1] >= 0 and pos[1] < map_size[1] and pos not in obstacles:
                    stack.append(pos)
    return False

def randAgents1(map_size, num_agents, num_groups, num_obstacles):
    locations = [(x, y) for x in range(0, map_size[0]) for y in range(0, map_size[1])]

    random.shuffle(locations)

    #
    Group = collections.namedtuple('Group', 'start goal')
    groups = []
    obstacles = []

    # assign obstacles
    for agentIdx in range(0, num_obstacles):
        location = locations[0]
        obstacles.append(location)
        del locations[0]

    locationsE = copy.deepcopy(locations) #list(locations)
    random.shuffle(locationsE)

    # different number of agents; fixed agents per group
    for groupIdx in range(0, num_groups):
        group = Group(start=[], goal=[])
        groups.append(group)

    for agentIdx in range(0, num_agents):
        groupIdx = agentIdx % num_groups

        while True:
            locationS = locations[0]
            locationE = locationsE[0]

            if reachable(map_size, locationS, locationE, obstacles):
                groups[groupIdx].start.append(locationS)
                groups[groupIdx].goal.append(locationE)
                del locations[0]
                del locationsE[0]
                # print("reachable!")
                break
            else:
                # print("not reachable!")
                random.shuffle(locations)
                random.shuffle(locationsE)
                # try again...

    return groups, obstacles

def writeFile(obstacles, map_size, groups, file_name):
    data = dict()
    data["map"] = dict()
    data["map"]["dimensions"] = map_size
    data["map"]["obstacles"] = [list(o) for o in obstacles]
    data["agents"] = []
    i = 0
    for group in groups:
        for agentIdx in range(0, len(group.start)):
            agent = dict()
            agent["name"] = "agent" + str(i)
            agent["start"] = list(group.start[agentIdx])
            agent["goal"] = list(group.goal[agentIdx])
            i += 1
            data["agents"].append(agent)
    with open(file_name, "w") as f:
        yaml.dump(data, f, indent=4, default_flow_style=None)

if __name__ == "__main__":

    # map_size = [32, 32]
    map_size = [8, 8]
    num_agents = 15
    # num_groups = num_agents
    num_obstacles = int(map_size[0] * map_size[1] * 0.2)

    for num_agents in [1]:
    # for num_agents in range(15, 21):
    # for num_agents in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        for i in range(0, 10):
          groups, obstacles = randAgents1(map_size, num_agents, num_agents, num_obstacles)
          writeFile(obstacles, map_size, groups, "map_{0}by{1}_obst{2}_agents{3}_ex{4:04}.yaml".format(
              map_size[0],
              map_size[1],
              num_obstacles,
              num_agents,
              i))
