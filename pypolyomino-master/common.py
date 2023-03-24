from constraint import Problem
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union
from svgwrite import Drawing
from svgwrite.path import Path


class Board(object):
    """
    holder object so that global variables do not need to be used
    """

    def __init__(self, width, length, shapes, unique=True, margin=True):
        self.shapes = shapes
        # if True, there cannot be more than one copy of a piece on a board
        self.unique = unique
        # add one square of margin for the edges of the board - not sure this is necessary,
        # but it might provide a speedup
        self.width = width
        self.margin = margin
        self.l1 = length + margin  # bottom margin of 1 cube
        self.w1 = width + margin  # right margin of 1 cube
        self.l2 = length + 2 * margin
        self.w2 = width + 2 * margin
        self.length = length
        self.board = [None for _ in range(0, self.w2 * self.l2)]
        self.used = [False for _ in range(0, len(set(shape[0] for shape in shapes)))]
        self.solution = []

        if not margin:
            return
        # mark the edges of the board as unavailable
        for i in range(0, self.w2):
            self.board[i] = -1
            self.board[self.w2 * self.l1 + i] = -1

        for i in range(0, self.l2):
            self.board[self.w2 * i] = -1
            self.board[self.w2 * i + self.w1] = -1

    # flip through width and look for asymmetry
    def wflip(self):
        for i in range(0, self.l2):
            for j in range(0, 3):
                d = self.board[i * self.w2 + j] - self.board[i * self.w2 + self.w1 - j]
                if not d:
                    continue
                return d > 0

    def lflip(self):
        for i in range(1, self.w2):
            for j in range(1, 8):
                d = self.board[j * self.w2 + i] - self.board[(self.l1 - j) * self.w2 + i]
                if not d:
                    continue
                return d > 0

    def hsplit(self):
        """
        Horizontal split, only applicable on the standard board.
        The count comes out 16, but the top piece, having the cross on its left,
        can be reflected vertically,
        while the bottom can be reflected or rotated, thus 8 combinations.
        There are really only 2 split solutions as follows.

        	EEEIIBJJJJ
        	EAEIIBGJHH
        	AAALIBGGGH
        	CALLLBDFGH
        	CKKKLBDFFH
        	CCCKKDDDFF

        	EEEIIBJJJJ
        	EAEIIBGJHH
        	AAAKIBGGGH
        	CALKKBDFGH
        	CLLLKBDFFH
        	CCCLKDDDFF

        """
        for i in range(1, self.w1):
            if self.board[5 * self.w2 + i] == self.board[6 * self.w2 + i]:
                return 0
            return 1

    def print_board_locations(self):
        for col in range(self.margin, self.w1):
            for row in range(self.margin, self.l1):
                print(str(self.w2 * row + col) + ",", end="")
            print('')
        print()

    def board_index_to_row_col(self, board_index):
        row = int(board_index / self.w2)
        col = board_index % self.w2
        return row, col

    def board_row_col_to_index(self, row, col):
        return self.w2 * row + col

    def print_board(self):
        for col in range(self.margin, self.w1):
            for row in range(self.margin, self.l1):
                piece_num = self.board[self.w2 * row + col]
                if piece_num is not None and piece_num < 0:
                    raise ValueError(f"got bad character! {piece_num}")
                _char = chr(piece_num + ord('A')) if piece_num is not None else '-'
                print(_char, end="")
            print('')
        print()

    def place_on_board(self, piece_index, loc):
        piece = self.shapes[piece_index][0]
        self.used[piece] = True

        self.board[loc] = piece
        for i in range(1, len(self.shapes[piece_index])):
            self.board[loc + self.shapes[piece_index][i]] = piece
        self.solution.append((piece_index, loc))

    def remove_piece_from_board(self, piece_index, loc):
        piece = self.shapes[piece_index][0]
        self.used[piece] = False
        self.board[loc] = None
        for i in range(1, len(self.shapes[piece_index])):
            self.board[loc + self.shapes[piece_index][i]] = None
        sol_piece_index, sol_loc = self.solution.pop()
        # if we didn't remove a copy of that piece from the solution, something bad happened
        assert sol_piece_index == piece_index
        assert sol_loc == loc

    def findloc(self):
        for i in range(self.w2 + 1, self.w2 * self.l1 - 1):
            if self.board[i] is None:
                return i
        return None

    def board_has_holes(self):
        for i in range(self.w2 + 1, self.w2 * self.l1 - 1):
            if self.board[i] is None:
                row, col = self.board_index_to_row_col(i)
                neighbors = [self.board[self.board_row_col_to_index(row + 1, col)],
                             self.board[self.board_row_col_to_index(row - 1, col)],
                             self.board[self.board_row_col_to_index(row, col - 1)],
                             self.board[self.board_row_col_to_index(row, col + 1)]]
                if all(neighbors):
                    return 0
        return 1

    def is_hole(self, loc):
        if self.board[loc] is not None:
            return 1
        row, col = self.board_index_to_row_col(loc)
        neighbors = [self.board[self.board_row_col_to_index(row + 1, col)],
                     self.board[self.board_row_col_to_index(row - 1, col)],
                     self.board[self.board_row_col_to_index(row, col - 1)],
                     self.board[self.board_row_col_to_index(row, col + 1)]]
        if all(neighbors):
            return 0
        return 1

    def test(self, loc, pattern):
        piece = self.shapes[pattern][0]
        if self.used[piece] and self.unique:
            return 0
        for shape_loc in self.shapes[pattern][1:]:
            if self.board[loc + shape_loc] is not None:
                return 0
        # we also want to make sure that the board does not contain any empty islands
        return 1

    def print_shapes(self):
        self.print_board_locations()
        for shape in self.shapes:
            loc = self.w2 + 3
            self.place_on_board(shape[0], loc)
            print(shape)
            self.print_board()
            self.remove_piece_from_board(shape[0], loc)


def rebuild_shapes(board_object, margin=True):
    for shape in board_object.shapes:
        for j in range(1, len(shape)):
            k = shape[j]
            if margin:
                k += 3
            row = int(k / 8)
            col = k % 8
            if margin:
                col -= 3
            k = board_object.w2 * row + col
            shape[j] = k


def output_to_svg(board_object, nsols):
    square_size = 40
    margin = 4
    filename = f"w{board_object.width}sol{nsols}.svg"
    drawing = Drawing(filename)
    colors = [
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod"
    ]
    pieces = []

    for piece_index, loc in board_object.solution:
        polygons = []
        for square_loc in [0] + board_object.shapes[piece_index][1:]:
            board_loc = loc + square_loc
            row, col = board_object.board_index_to_row_col(board_loc)
            rect_points = [(square_size * row, square_size * col), (square_size * (row + 1), square_size * col),
                           (square_size * (row + 1), square_size * (col + 1)),
                           (square_size * row, square_size * (col + 1))]
            polygons.append(Polygon(rect_points))
        polygon = cascaded_union(polygons)
        pieces.append(polygon)

    # get a coloring for the shapes solving the four color problem as a CSP
    problem = Problem()
    for i in range(len(pieces)):
        problem.addVariable(i, [0, 1, 2, 3])
        for j in range(len(pieces)):
            if i == j:
                continue
            if pieces[i].intersects(pieces[j]):
                problem.addConstraint(lambda x, y: x != y, (i, j))
    # there should always be at least one solution
    coloring_solution = problem.getSolution()
    for i, piece in enumerate(pieces):
        if isinstance(piece, MultiPolygon):
            raise ValueError(
                f"bad solution! {board_object.solution}, a piece was probably placed off the edge of the board")
        piece_points = piece.buffer(-margin).exterior.coords
        d = f"M {piece_points[0][0]} {piece_points[0][1]} "
        for x, y in piece_points[1:]:
            d += f"L {x} {y} "
        shape_index = board_object.solution[i][0]
        _id = f"{shape_index}_{board_object.shapes[shape_index][0]}"
        drawing.add(Path(d=d, fill=colors[int(coloring_solution[i])], id=_id))

    drawing.save(pretty=2)
