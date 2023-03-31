from typing import List, Tuple
import cv2

# 큰 직사각형 정의
MAX_WIDTH = 1000
MAX_HEIGHT = 1000
rectangles = [(100, 30), (40, 60), (30, 30), (70, 70), (100, 50), (30, 30)]
def calculate_rectangle_area(rectangle: Tuple[int, int]) -> int:
    """
    직사각형의 면적을 계산하는 함수
    Args:
        rectangle: (width, height) 형태의 튜플
    Returns:
        직사각형의 면적
    """
    return rectangle[0] * rectangle[1]
def sort_rectangles(rectangles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    작은 직사각형들을 면적 내림차순으로 정렬해주는 함수
    Args:
        rectangles: (width, height) 형태의 튜플 리스트
    Returns:
        면적이 내림차순으로 정렬된 작은 직사각형 리스트
    """
    return sorted(rectangles, key=calculate_rectangle_area, reverse=True)
def place_small_rectangle(grid: List[List[int]], x: int, y: int, width: int, height: int, index: int) -> None:
    """
    작은 직사각형을 해당 위치에 배치하는 함수
    Args:
        grid: 큰 직사각형 리스트
        x: 작은 직사각형이 배치될 x 좌표
        y: 작은 직사각형이 배치될 y 좌표
        width: 작은 직사각형의 너비
        height: 작은 직사각형의 높이
        index: 작은 직사각형 리스트에서의 인덱스
    """
    for i in range(x, x + width):
        for j in range(y, y + height):
            grid[i][j] = index
def place_all_rectangles(grids: List[List[List[int]]], rectangles: List[Tuple[int, int]]) -> int:
    """
    모든 작은 직사각형들을 큰 직사각형에 배치하는 함수
    Args:
        grids: 큰 직사각형 리스트
        rectangles: (width, height) 형태의 튜플 리스트
    Returns:
        큰 직사각형의 갯수
    """
    sorted_rectangles = sort_rectangles(rectangles)
    # 크기별로 정렬된 작은 직사각형을 하나씩 가져와서 큰 직사각형에 배치함
    for rectangle_index, (width, height) in enumerate(sorted_rectangles):
        rectangle_placed = False
        for grid_index, grid in enumerate(grids):
            for x in range(MAX_WIDTH - width + 1):
                for y in range(MAX_HEIGHT - height + 1):
                    # 가로방향 배치
                    if is_location_empty(grid, x, y, width, height):
                        place_small_rectangle(grid, x, y, width, height, rectangle_index)
                        rectangle_placed = True
                        break
                    # 90도 회전 후 가로방향 배치
                    elif is_location_empty(grid, x, y, height, width):
                        place_small_rectangle(grid, x, y, height, width, rectangle_index, True)
                        rectangle_placed = True
                        break
                if rectangle_placed:
                    break
                # 아랫줄로 이동 후 가로방향 배치
                if y == MAX_HEIGHT - height and is_location_empty(grid, x, y + 1, width, height):
                    place_small_rectangle(grid, x, y + 1, width, height, rectangle_index)
                    rectangle_placed = True
                    break
                # 90도 회전 후 아랫줄로 이동 후 가로방향 배치
                elif y == MAX_HEIGHT - height and is_location_empty(grid, x, y + 1, height, width):
                    place_small_rectangle(grid, x, y + 1, height, width, rectangle_index, True)
                    rectangle_placed = True
                    break
            if rectangle_placed:
                break

        # 새로운 큰 직사각형을 만듦
        if not rectangle_placed:
            grids.append([[0] * MAX_WIDTH for _ in range(MAX_HEIGHT)])
            place_all_rectangles(grids, rectangles[rectangle_index:])
            break

    # 큰 직사각형의 갯수를 반환함
    return len(grids)
