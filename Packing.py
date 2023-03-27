from typing import List, Tuple

def pack_rectangles(rectangles: List[Tuple[int, int]]) -> int:
    def pack(subrectangles: List[Tuple[int, int]]) -> Tuple[int, int]:
        # 패킹된 직사각형의 폭과 높이를 반환하는 함수
        width = sum(r[0] for r in subrectangles)
        height = max(r[1] for r in subrectangles)
        return width, height

    def rotate(rectangle: Tuple[int, int]) -> Tuple[int, int]:
        # 직사각형을 회전시킨 결과를 반환하는 함수
        return rectangle[1], rectangle[0]

    rectangles = [tuple(sorted(rect)) for rect in rectangles]
    rectangles.sort(key=lambda r: r[0], reverse=True)

    packed_rectangles = []
    while rectangles:
        # 현재 패킹중인 작은 직사각형 리스트
        subrectangles = [rectangles.pop(0)]
        # 패킹된 직사각형의 크기를 계산
        width, height = pack(subrectangles)

        # 빈 공간을 채우는 작은 직사각형 리스트
        remaining_rectangles = []
        for rect in rectangles:
            if rect[0] <= height and width + rect[1] <= max(r[1] for r in subrectangles + remaining_rectangles):
                # 작은 직사각형을 패킹할 수 있는 경우
                subrectangles.append(rect)
                width, height = pack(subrectangles)
            elif rect[1] <= height and width + rect[0] <= max(r[1] for r in subrectangles + remaining_rectangles):
                # 작은 직사각형을 회전시켜 패킹할 수 있는 경우
                subrectangles.append(rotate(rect))
                width, height = pack(subrectangles)
            else:
                # 작은 직사각형을 패킹할 수 없는 경우
                remaining_rectangles.append(rect)

        packed_rectangles.append(subrectangles)
        rectangles = remaining_rectangles

    return len(packed_rectangles)
