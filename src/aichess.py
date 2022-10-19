import chess
import sys
import queue
import numpy as np  # np.sqrt(n)

# RawStateType = List[List[List[int]]]

from itertools import permutations
import copy


class State:
    def __init__(self, state_list):
        self.list = state_list
        self.set = self.setify(state_list)
        self.visited = True

class Aichess():
    """
    A class to represent the game of chess.
    ...
    Attributes:
    -----------
    chess : Chess
        represents the chess game
    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece
    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8;
        self.checkMate = False
        self.checkmateStates = [{(0, 0, 2), (2, 4, 6)}, {(0, 1, 2), (2, 4, 6)}, {(0, 2, 2), (2, 4, 6)},
                                {(0, 6, 2), (2, 4, 6)},{(0, 7, 2), (2, 4, 6)}]

    def setify(self, states):
        """
        function that converts from list to frozenset
        Args:
            states: list [[0,0,2],[2,4,6]]
        Returns: frozenset{(0,0,2),(2,4,6)}
        """
        setStates = set()

        for state in states:
            tup = tuple(state)
            setStates.add(tup)

        return frozenset(setStates)

    def getCurrentState(self):
        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def isCheckMate(self, mystate):
        """
        returns True if the current state matches any checkmate state in the checkmate list
        Args:
            mystate: current state of board (set)
        Returns:
            True: we are in check mate state
            False: we are not in checkmate state
        """
        if mystate in self.checkmateStates:
            return True
        return False

    def DepthFirstSearch(self, currentState, depth):
        """
        Check mate from currentStateW through DFS backtracking
        """
        # Declaration of variables:
        # result - store the final path
        # visited - explored moves
        result = [currentState]
        visited = set()

        # Recursive DFS.
        def dfs_backtracking(currentState, depth, result):
            # base cases:
            # 1.Finish when check-mate
            # 2.Finish when have check all possibilities

            # Setify the state list and add it to visited.
            current = self.setify(currentState)
            visited.add(current)

            # Check if the current state is a Checkmate.
            if self.isCheckMate(current):
                print('\nCheckmate!! \nAt ',list(current)," and length path ",depth)
                return True

            # 1. Iterate all possible states from currentState, bearing in mind that once we have
            # exceeded the maxDepth, the path is exceeding the limit of states and return.
            if depth < self.depthMax:
                for state in self.getListNextStatesW(currentState):
                    setified_state = self.setify(state)

                    # For each possible state, if it hasn't been visited or is the same as the current one, we explore it.
                    if setified_state not in visited and not self.isSameState(state,currentState):
                        result.append(state) # Add it to the current path.
                        # self.move(self.chess, current,setified_state) * Details in the report.

                        # Recursive call with the state to be explored and its depth.
                        if dfs_backtracking(state, depth + 1, result):
                            # If it returns True, we found a Checkmate state and update the pathToTarget with the current path.
                            self.pathToTarget = result
                            return True

                        # Backtracking of the variables.
                        result.pop()
                        visited.remove(setified_state)
                        # self.move(self.chess, setified_state, current) * Details in the report.

            # 2. We have iterated through all the possibilities and still can't find check-mate state.
            return False

        return dfs_backtracking(currentState, depth, result)

    def rebuild_path(self, prev, destination, depth):
        """
        function that rebuild the path with the help of the dictionary prev
        Args:
            prev: dictionary with the previous states
        Returns:
            list[path]
        """

        # The first item in the list would be the checkmate state we reached.
        path = [list(destination)]

        # We build the path with the previous destination.
        for i in range(0,depth):
            path.append(list(prev[destination]))
            destination = prev[destination]

        return path

    def BreadthFirstSearch(self, currentState, depth):
        """
        Check mate from currentStateW through BFS
        """
        # Setify currentState (list of lists) to current frozenset.
        current = self.setify(currentState)

        # Dictionary in which we will save the previous states and then to reconstruct the path.
        prev = dict()
        prev[current] = None

        # Define a queue to explore the tree in FIFO and add the root (currentState).
        que = queue.Queue()
        que.put((current, depth))

        # List of explored states, set instead of list to enhance efficiency.
        visit = set()
        visit.add(current)

        while not que.empty():
            # Get the first element on the queue.
            data = que.get()

            # Retrieve the state data:
            # State is the current state
            # Depth is the number of the level we have expanded so far
            state, depth = data[0], data[1]

            # Check if the actual state is a checkmate, in that case, return.
            if self.isCheckMate(state):
                print('\nCheckmate!! \nAt ', list(state), " and length path ", depth)
                rebuilded = self.rebuild_path(prev, state, depth)
                self.pathToTarget = rebuilded[::-1]
                return

            # If we're within the limits of depth, explore the current state.
            if depth < self.depthMax:
                for nextState in self.getListNextStatesW(list(state)):
                    setified_nextState = self.setify(nextState) # Setify the next state.

                    # If it hasn't been visited nor is the same state as the current one, we put it in the queue.
                    if setified_nextState not in visit and not self.isSameState(nextState, list(state)):
                        visit.add(setified_nextState) # Add it to the visited set.
                        prev[setified_nextState] = state # Assign its father.
                        que.put((setified_nextState, depth + 1))  # Put it in the queue.

    def compute_heuristic(self, currentState):
        """
        Function that calculates the minimum distance between the current state and all checkmate states in checkmateList.
        """

        current = list(currentState)

        bestDist = float("inf") # Initialize the best distance.

        # For every checkmate, compare its states with both of the current pieces.
        for checkmate in self.checkmateStates:
            dist = 0
            for state in checkmate:
                checkmate_x = state[0]
                checkmate_y = state[1]

                # For both King and Rock, we calculate the sum of the two distances.
                for piece in current:
                    if state[2] == piece[2]:
                        piece_x = piece[0]
                        piece_y = piece[1]
                        dist += abs((piece_x - checkmate_x)) + abs((piece_y - checkmate_y))

            # If the current best is greater than the current total, update the best.
            if dist < bestDist:
                bestDist = dist

        return bestDist

    def AStarSearch(self, currentState, depth):
        """
        Check mate from currentStateW through A*
        Args:
            currentState:
            depth:
        Returns:
        """
        current = self.setify(currentState) # Setify the current state for efficiency.

        # Dictionary in which we will save the previous states and then to reconstruct the path.
        prev = dict()
        prev[current] = None

        # Define a priority queue to explore the tree depending on costs.
        que = queue.PriorityQueue()
        que.put((0, current, depth))

        # List of explored states, set instead of list to enhance efficiency.
        visit = set()
        visit.add(current)

        # Dictionary of each state cost.
        costs = dict()
        costs[current] = 0

        while que:
            # Get the element with top priority (minimum cost).
            data = que.get()

            # Retrieve the state data:
            # State is the current state
            # Depth is the number of the level we have expanded so far
            state, depth = data[1], data[2]

            # Check if the actual state is a checkmate, in that case, return.
            if self.isCheckMate(state):
                print('\nCheckmate!! \nAt ', list(state), " and length path ", depth)
                rebuilded = self.rebuild_path(prev, state, depth)
                self.pathToTarget = rebuilded[::-1]
                return

            # If we're within the limits of depth, explore the current state.
            if depth < self.depthMax:
                for nextState in self.getListNextStatesW(list(state)):
                    setified_nextState = self.setify(nextState)  # Setify the next state.

                    # Calculate the cost of the next state.
                    # The value will be the current depth + the heuristic result.
                    cost = depth + self.compute_heuristic(nextState)

                    # If it hasn't been visited nor is the same state as the current one, we put it in the queue.
                    if setified_nextState not in visit and not self.isSameState(nextState, list(state)):
                        visit.add(setified_nextState) # Add it to the visited set.
                        prev[setified_nextState] = state # Assign its father.
                        que.put((cost,setified_nextState, depth + 1)) # Put it in the queue specifying its priority.
                        costs[setified_nextState] = cost # Store the cost to compare.

                    # If the next state has already been visited, check if the newest cost is lesser than the stored one.
                    elif costs[setified_nextState] > cost:
                        prev[setified_nextState] = state # Update the father.
                        costs[setified_nextState] = cost # Update the cost.


    def minimax(self, currentState, depth, black):

        # Si hem superat el depth definit o l'estat actual és un checkmate, avaluem la posició.
        if depth==0 or self.isCheckmate(currentState.set):
            return self.compute_heuristic(self, currentState.set)

        # Si estan jugant les negres, escollim el cost més gran avaluat. (heurística major, menys proper a un checkmate blanc)
        if black:
            maxCost = -float('inf')
            for nextState in self.getListNextStatesW(currentState.list):
                if not isinstance(nextState,State) or not nextState.visited:
                    nextState = State(nextState)
                    cost = self.minimax(nextState,depth-1,False)
                    maxCost = max(maxCost, cost)
                    #print("Current Black, move: ", nextState, ", cost: ", cost, "selected: ", maxCost)
            return maxCost

        # Si estan jugant les blanques, escollim el cost més petit avaluat. (heurística menor, més proper a un checkmate blanc)
        else:
            minCost = +float('inf')
            for nextState in self.getListNextStatesW(currentState.list):
                if not isinstance(nextState, State) or not nextState.visited:
                    nextState = State(nextState)
                    cost = self.minimax(nextState, depth - 1, False)
                    minCost = min(minCost, cost)
                    #print("Current WHITE, move: ",nextState,", cost: ",cost,"selected: ",minCost)
            return minCost



    def tests(self, currentState):
        '''
        if self.isCheckMate(currentState):
            return True
        return False
        print("nextStates of [[0, 0, 2], [7, 4, 6]]: ", self.getListNextStatesW(state))
        state = [[7, 0, 2], [7, 4, 6]]
        state1 = [[0,0,2],[7,4,6]]
        manhattan = self.compute_manhattan(state,state1)
        print('the manhatan distance is:',manhattan)
        print('isCheckmate: ',self.isCheckMate(currentState))
         state = self.setify(currentState)
        print('ischeckmate:',self.isCheckMate(state))
        state = [[7, 0, 2], [7, 4, 6]]
        state1 = [[0, 0, 2], [7, 4, 6]]
        state = self.setify(state)
        manhattan = self.compute_manhattan(self.setify(state), self.setify(state1))
        print('the manhatan distance is:', manhattan)
        perm_state = list(permutations(self.checkmateStates))
        print('perm_state:',perm_state)
        print('not permur_states:',self.checkmateStates)
        '''

        state = [[0, 0, 2], [7, 4, 6]]
        print('la dist min entre[[0, 0, 2], [7, 4, 6]] i els estats checkmates es:', self.compute_heuristic(state))

    def move(self, chess_board, standard_current_state, standard_next_state):
        standard_current_state = list(standard_current_state)
        standard_next_state = list(standard_next_state)
        print('Format of stantard_current_state:',standard_current_state)
        print('Format of standar_nextState:',standard_next_state)
        start = [e for e in standard_current_state if e not in standard_next_state]
        to = [e for e in standard_next_state if e not in standard_current_state]
        start, to = start[0][0:2], to[0][0:2]

        print('format start: ',start)
        print('format end:', to)
        # this line may vary
        self.chess.moveSim(start, to)


def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """
    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None


def get_board_details():
    # get list of next states for current state
    print("current State", currentState)

    # it uses board to get them... careful
    aichess.getListNextStatesW(currentState)
    print("list next states ", aichess.pathToTarget)

    # starting from current state find the end state (check mate) - recursive function
    # find the shortest path, initial depth 0
    depth = 0
    # aichess.DepthFirstSearch(currentState, depth)
    print("DFS End")

    # example move piece from start to end state
    MovesToMake = ['1e', '2e']
    print("start: ", MovesToMake[0])
    print("to: ", MovesToMake[1])

    start = translate(MovesToMake[0])
    to = translate(MovesToMake[1])

    print("start: ", start)
    print("to: ", to)

    aichess.chess.moveSim(start, to)

    aichess.chess.boardSim.print_board()
    print("#Move sequence...  ", aichess.pathToTarget)
    print("#Visited sequence...  ", aichess.listVisitedStates)
    print("#Current State...  ", aichess.chess.board.currentStateW)


if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))
    TA[7][0] = 2
    TA[7][7] = 6
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()

    depth = 0
    aichess.DepthFirstSearch(currentState, depth)
    print("DFS: pathToTarget:", aichess.pathToTarget)
    print("DFS End\n")

    aichess.BreadthFirstSearch(currentState, depth)
    print("BFS: pathToTarget:", aichess.pathToTarget)
    print("BFS End\n")

    aichess.AStarSearch(currentState,depth)
    print("A*: pathToTarget:", aichess.pathToTarget)
    print("A* End")

    #get_board_details()
    #aichess.tests([[2, 4, 6], [0, 0, 2]])