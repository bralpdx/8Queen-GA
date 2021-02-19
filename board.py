"""
Referenced: https://kibiwebgeek.com/create-a-chessboard-in-python-and-pygame/
And added representation of Queen placement via Blue Circles.
"""

import sys, pygame
from n_queens import N

pygame.init()


class Board:
    def __init__(self, solution=[]):
        self.solution = solution
        self.__run()

    def __run(self):
        if len(self.solution) <= 1:
            assert False, "No solution found."

        size = width, height = 512, 512
        black = 0, 0, 0
        white = 255, 255, 255
        blue = 0, 0, 225
        title = "8Queens by Jonathan Bral"

        screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)

        list_of_queens = list()
        for q in range(len(self.solution)):
            xy = [q, self.solution[q]]
            list_of_queens.append(xy)

        rect_list = list()
        # used this loop to create a list of rectangles
        for i in range(0, N):  # control the row
            for j in range(0, N):  # control the column
                if i % 2 == 0:  # which means it is an even row
                    if j % 2 != 0:  # which means it is an odd column
                        rect_list.append(pygame.Rect(j * 64, i * 64, 64, 64))
                else:
                    if j % 2 == 0:  # which means it is an even column
                        rect_list.append(pygame.Rect(j * 64, i * 64, 64, 64))

        # create main surface and fill the base color with light brown color
        chess_board_surface = pygame.Surface(size)
        chess_board_surface.fill(white)

        # next draws black rectangles on the chessboard surface
        for chess_rect in rect_list:
            pygame.draw.rect(chess_board_surface, black, chess_rect)

        for q in list_of_queens:
            xy = [q[0] * 64 + 32, q[1] * 64 + 32]
            pygame.draw.circle(chess_board_surface, blue, xy, 16)

        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            # displayed the chess surface
            screen.blit(chess_board_surface, (0, 0))
            pygame.display.flip()


if __name__ == "__main__":
    # Used for testing purposes
    temp = [6, 1, 5, 2, 0, 3, 7, 4]
    game = Board(temp)
