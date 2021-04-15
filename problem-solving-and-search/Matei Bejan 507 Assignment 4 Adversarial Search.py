class TicTacToe:
    def __init__(self, start):
        self.state = [['.', '.', '.'], ['.', '.', '.'], ['.', '.', '.']]
        if start == 'player':
            self.player_turn = 'X'
        elif start == 'machine':
            self.player_turn = 'O'

    def draw_board(self):
        for i in range(0, 3):
            for j in range(0, 3):
                print('{}|'.format(self.state[i][j]), end=' ')
            print()
        print()

    def is_valid_move(self, x, y):
        if x < 0 or x > 2 or y < 0 or y > 2:
            return False
        elif self.state[x][y] != '.':
            return False
        else:
            return True

    def has_game_ended(self):
        for i in range(0, 3):
            if (self.state[0][i] != '.' and
                self.state[0][i] == self.state[1][i] and
                self.state[1][i] == self.state[2][i]):
                return self.state[0][i]

        for i in range(0, 3):
            if (self.state[i] == ['X', 'X', 'X']):
                return 'X'
            elif (self.state[i] == ['O', 'O', 'O']):
                return 'O'

        if (self.state[0][0] != '.' and
            self.state[0][0] == self.state[1][1] and
            self.state[0][0] == self.state[2][2]):
            return self.state[0][0]

        if (self.state[0][2] != '.' and
            self.state[0][2] == self.state[1][1] and
            self.state[0][2] == self.state[2][0]):
            return self.state[0][2]

        for i in range(0, 3):
            for j in range(0, 3):
                if (self.state[i][j] == '.'):
                    return None

        return '.'
    
    def max(self):
        maxi, x, y = -2, None, None
        result = self.has_game_ended()

        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)

        for i in range(0, 3):
            for j in range(0, 3):
                if self.state[i][j] == '.':
                    self.state[i][j] = 'O'
                    m, min_i, min_j = self.min()
                    if m > maxi:
                        maxi, x, y = m, i, j
                    self.state[i][j] = '.'

        return maxi, x, y

    def min(self):
        mini, x, y = 2, None, None
        result = self.has_game_ended()

        if result == 'X':
            return (-1, 0, 0)
        elif result == 'O':
            return (1, 0, 0)
        elif result == '.':
            return (0, 0, 0)

        for i in range(0, 3):
            for j in range(0, 3):
                if self.state[i][j] == '.':
                    self.state[i][j] = 'X'
                    m, max_i, max_j = self.max()
                    if m < mini:
                        mini, x, y = m, i, j
                    self.state[i][j] = '.'

        return mini, x, y

    def play(self):
        first_round = True
        while True:
            self.draw_board()
            self.result = self.has_game_ended()

            if self.result != None:
                if self.result == 'X':
                    print('The player wins!')
                elif self.result == 'O':
                    print('The machine wins!')
                elif self.result == '.':
                    print("Tie!")

                return

            if self.player_turn == 'X':

                while True:
                    m, x, y = self.min()

                    if first_round:
                        print('Recommended move: X = {}, Y = {}.'.format(1, 1))
                        first_round = False
                    else:
                        print('Recommended move: X = {}, Y = {}.'.format(x, y))

                    x = int(input('Insert X coordinate: '))
                    y = int(input('Insert Y coordinate: '))

                    if self.is_valid_move(x, y):
                        self.state[x][y] = 'X'
                        self.player_turn = 'O'
                        break
                    else:
                        print('Invalid move! Please insert valid coodrinates.')
            else:
                m, x, y = self.max()
                self.state[x][y] = 'O'
                self.player_turn = 'X'

ttt = TicTacToe('machine')
ttt.play()
