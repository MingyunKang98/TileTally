# hexsol.py  - a translation of Karl Dahlke's polyomino packing program to python
# not for commercial use
from argparse import ArgumentParser

# lay out shapes on an 8x8 board.
from common import Board, rebuild_shapes, output_to_svg

shapes = [[0, 7, 8, 9, 16],
          [1, 1, 2, 3, 4], [1, 8, 16, 24, 32],
          [2, 1, 2, 8, 16], [2, 1, 2, 10, 18], [2, 8, 14, 15, 16], [2, 8, 16, 17, 18],
          [3, 1, 2, 9, 17], [3, 6, 7, 8, 16], [3, 8, 15, 16, 17], [3, 8, 9, 10, 16],
          [4, 1, 2, 8, 10], [4, 1, 9, 16, 17], [4, 2, 8, 9, 10], [4, 1, 8, 16, 17],
          [5, 1, 7, 8, 15], [5, 1, 9, 10, 18], [5, 7, 8, 14, 15], [5, 8, 9, 17, 18],
          [6, 1, 8, 15, 16], [6, 8, 9, 10, 18], [6, 1, 9, 17, 18], [6, 6, 7, 8, 14],
          [7, 1, 2, 3, 8], [7, 1, 9, 17, 25], [7, 5, 6, 7, 8], [7, 8, 16, 24, 25], [7, 8, 9, 10, 11], [7, 1, 8, 16, 24],
          [7, 1, 2, 3, 11], [7, 8, 16, 23, 24],
          [8, 1, 2, 8, 9], [8, 1, 8, 9, 17], [8, 1, 7, 8, 9], [8, 8, 9, 16, 17], [8, 1, 8, 9, 10], [8, 1, 8, 9, 16],
          [8, 1, 2, 9, 10], [8, 7, 8, 15, 16],
          [9, 1, 2, 3, 9], [9, 7, 8, 16, 24], [9, 6, 7, 8, 9], [9, 8, 16, 17, 24], [9, 1, 2, 3, 10], [9, 8, 15, 16, 24],
          [9, 7, 8, 9, 10], [9, 8, 9, 16, 24],
          [10, 1, 9, 10, 11], [10, 7, 8, 15, 23], [10, 1, 2, 10, 11], [10, 8, 15, 16, 23], [10, 1, 2, 7, 8],
          [10, 8, 9, 17, 25], [10, 1, 6, 7, 8], [10, 8, 16, 17, 25],
          [11, 1, 9, 10, 17], [11, 6, 7, 8, 15], [11, 7, 8, 16, 17], [11, 7, 8, 9, 15], [11, 1, 7, 8, 16],
          [11, 7, 8, 9, 17], [11, 8, 9, 15, 16], [11, 8, 9, 10, 17],
          ]

'''
valid positions for the cross
prevents reflections and rotations
But extra code is needed if the width or height is odd
'''
cross_all = [
    [12, 17, 22, 27, 32, 37, 42, 47, 0],
    [14, 20, 26, 32, 38, 44, 0],
    [10, 16, 17, 23, 24, 30, 31, 37, 38, 0],
    [11, 18, 19, 26, 27, 34, 35, 0],
    [0],
    [13, 14, 23, 0]
]


# position the cross, and go
def cross(board_object, top, nsols):
    board_object.place_on_board(piece_index=0, loc=top)
    nsols = place(board_object, nsols=nsols)

    board_object.remove_piece_from_board(piece_index=0, loc=top)
    return nsols


# place a piece in the board; recursive
def place(board_object, nsols):
    # find best location
    loc = board_object.findloc()
    if not loc:
        return  # square that no piece fits into

    piece_index = 1
    while piece_index < len(shapes):
        piece_type = shapes[piece_index][0]
        if not board_object.test(loc, piece_index):
            piece_index += 1
            continue

        # some checks for symmetry for odd dimensions
        if loc == 21 and args.width == 3 and piece_type == 8:
            piece_index += 1
            continue

        #  place the piece
        piece = shapes[piece_index][0]

        board_object.place_on_board(loc=loc, piece_index=piece_index)
        if args.debug:
            print(f"placing piece {piece}[{piece_index}] at square {loc}, used {sum(used)}")

        if all(board_object.used):
            if (wcenter and board_object.wflip()) or (lcenter and board_object.lflip()) \
                    or (ocenter and board_object.board[13] > board_object.board[31]):
                #  skip this one
                pass
            else:
                nsols += 1
                if args.svg:
                    output_to_svg(board_object, nsols)
                #  print solution
                if args.dispflag:
                    print(f"solution {nsols}: ")
                    board_object.print_board()
        else:
            nsols = place(board_object, nsols)
        #  remove piece
        board_object.remove_piece_from_board(piece_index, loc)
        piece_index += 1
    return nsols


#  place


if __name__ == "__main__":
    parser = ArgumentParser(description='Generate rectangular packings of polyominos.')
    parser.add_argument(dest='width', type=int, help="the board width", default=0)
    parser.add_argument('-d', dest='dispflag', action='store_true', help='print all board solutions')
    parser.add_argument('-c', dest='countflag', action='store_true',
                        help='print number of solutions')
    parser.add_argument('-v', '--verbose', dest='debug', action='store_true',
                        help='print debugging info')
    parser.add_argument('-s', dest='svg', action='store_true', help='save solution as svg')

    args = parser.parse_args()
    l = 8 if args.width == 8 else int(60 / args.width)
    _board = Board(args.width, l, shapes)

    if args.width == 8:
        _board.board[10 * 5 + 5] = 26
        _board.board[10 * 4 + 5] = 26
        _board.board[10 * 4 + 4] = 26
        _board.board[10 * 5 + 4] = 26

    # scale the location of the shapes based on the board width
    rebuild_shapes(_board)
    cross_pos = cross_all[args.width - 3]

    nsols = 0

    i = 0

    # only for debugging
    while cross_pos[i]:
        # args.width = 3 handled in place()
        lcenter = wcenter = ocenter = 0
        if args.width == 4 and i == 5:
            lcenter = 1
        if args.width == 5 and not (i & 1):
            wcenter = 1
        if args.width == 8 and i == 2:
            ocenter = 1
        cross(_board, top=cross_pos[i], nsols=nsols)
        i += 1

    if args.countflag:
        print(f"{nsols} solutions\n")
