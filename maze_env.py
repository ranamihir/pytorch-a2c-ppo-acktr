import numpy as np
import h5py
import os
import argparse
from queue import Queue
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.wrappers import FullyObsWrapper, ActionBonus

parser = argparse.ArgumentParser()
parser.add_argument('--num-mazes', metavar='NUM_MAZES', dest='num_mazes', help='number of mazes', required=False, \
                    type=int, default=10000)
parser.add_argument('--grid-size', metavar='GRID_SIZE', dest='grid_size', help='grid size', required=False, \
                    type=int, default=20)
parser.add_argument('--min-num-rooms', metavar='MIN_NUM_ROOMS', dest='min_num_rooms', help='min number of rooms', \
                    required=False, type=int, default=5)
parser.add_argument('--max-num-rooms', metavar='MAX_NUM_ROOMS', dest='max_num_rooms', help='max number of rooms', \
                    required=False, type=int, default=10)
args = parser.parse_args()

NUM_MAZES = args.num_mazes
GRID_SIZE = args.grid_size
MIN_NUM_ROOMS = args.min_num_rooms
MAX_NUM_ROOMS = args.max_num_rooms


class Room:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class MultiRoomEnv(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        super(MultiRoomEnv, self).__init__(
            grid_size=GRID_SIZE
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.grid_size or topY + sizeY >= self.grid_size:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

class MultiRoomEnvN6(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=MIN_NUM_ROOMS,
            maxNumRooms=MAX_NUM_ROOMS
        )

register(
    id='MiniGrid-Maze-v0',
    entry_point=lambda: FullyObsWrapper(MultiRoomEnvN6()),
    reward_threshold=1000.0
)

def BFS(grid, q, visited, paths):
    current_index = q.get()
    current_x, current_y = current_index[0], current_index[1]

    element = grid[current_x, current_y]
    visited[current_x, current_y] = 1

    if element == 9:
        return current_x, current_y

    for x in range(current_x-1, current_x+2):
        for y in range(current_y-1, current_y+2):
            if not (x == current_x and y == current_y) \
                and not (abs(x-current_x) + abs(y-current_y) > 1) \
                and x >- 1 and y >- 1 \
                and x < grid.shape[0] and y < grid.shape[1] \
                and grid[x,y] not in [2] \
                and (x,y) not in q.queue \
                and not visited[x,y]:
                paths[(x,y)] = (current_x, current_y)
                q.put((x,y))
    return BFS(grid, q, visited, paths)

def get_optimal_path(obs):
    '''
    Grid:
        1   -> Movable cells
        2   -> Wall
        4   -> Door
        9   -> Reward
        255 -> Agent

    Orientations:
        Right -> 0
        Down  -> 1
        Left  -> 2
        Up    -> 3
    '''
    paths = {}

    grid, orientation = obs[:,:,0].T, obs[:,:,1].T

    visited = np.zeros(grid.shape)

    initial_position = tuple(np.array(np.where(grid == 255)).ravel())
    start_queue = Queue()
    start_queue.put(initial_position)
    reward_position = BFS(grid, start_queue, visited, paths)
    keys = get_optimal_keys(grid, paths, orientation, reward_position, initial_position)

    return keys

def get_optimal_keys(grid, paths, orientation, reward_position, initial_position):
    steps = [reward_position]

    # Get list of steps to be taken
    start_position = reward_position
    while paths[start_position] != initial_position:
        next_step = paths[start_position]
        steps.append(next_step)
        start_position = next_step
    steps.append(initial_position)
    steps = steps[::-1]

    # Get list of keys corresponding to each step
    orientation_map = {
        (0,0): [],
        (0,1): ['RIGHT'],
        (0,2): ['RIGHT']*2,
        (0,3): ['LEFT'],
        (1,0): ['LEFT'],
        (1,1): [],
        (1,2): ['RIGHT'],
        (1,3): ['RIGHT']*2,
        (2,0): ['RIGHT']*2,
        (2,1): ['LEFT'],
        (2,2): [],
        (2,3): ['RIGHT'],
        (3,0): ['RIGHT'],
        (3,1): ['RIGHT']*2,
        (3,2): ['LEFT'],
        (3,3): []
    }

    keys = []
    current_position = steps[0]
    current_orientation = orientation[current_position]
    for i in range(len(steps)-1):
        if steps[i+1][0] - steps[i][0] == 1:
            keys.extend(orientation_map[(current_orientation, 1)])
            current_orientation = 1
        elif steps[i+1][0] - steps[i][0] == -1:
            keys.extend(orientation_map[(current_orientation, 3)])
            current_orientation = 3
        elif steps[i+1][1] - steps[i][1] == 1:
            keys.extend(orientation_map[(current_orientation, 0)])
            current_orientation = 0
        elif steps[i+1][1] - steps[i][1] == -1:
            keys.extend(orientation_map[(current_orientation, 2)])
            current_orientation = 2

        # Open Door
        if grid[steps[i+1]] == 4:
            keys.append('SPACE')

        keys.append('UP')
        current_position = steps[i+1]

    return keys

def save_object(data, filepath):
    with h5py.File(filepath, 'w') as hf:
        for i in range(len(data)):
            hf.create_dataset('maze_{}'.format(i), data=np.transpose(data[i], (0,3,1,2)))

def main():
    import sys
    import time

    # Load the gym environment
    env = gym.make("MiniGrid-Maze-v0")
    env.seed(0)

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # # Create a window to render into
    # renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        # print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            resetEnv()

        return obs

    maze_idx = 1
    all_mazes = []

    try:
        while maze_idx <= NUM_MAZES:
            # Get the current observations
            obs = env.observation(0)
            current_maze = obs[np.newaxis,:]

            # Take optimal steps
            optimal_path_keys = get_optimal_path(obs)
            for key in optimal_path_keys:
                obs = keyDownCb(key)
                if obs is not None:
                    current_maze = np.vstack((current_maze, obs[np.newaxis,:]))
                # env.render('human')
                # time.sleep(0.02)
            all_mazes.append(current_maze)

            print('Finished maze number {}.'.format(maze_idx))
            maze_idx += 1

    except KeyboardInterrupt:
        maze_idx -= 1
        pass

    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filepath = os.path.join(data_dir, 'all_mazes_{}_{}_{}_{}.h5'.format(min(maze_idx, NUM_MAZES), GRID_SIZE, MIN_NUM_ROOMS, MAX_NUM_ROOMS))
    save_object(all_mazes, filepath)


if __name__ == "__main__":
    main()
