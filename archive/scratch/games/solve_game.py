"""
兔兔和蛋蛋的棋类游戏 - 找出兔兔犯错的地方

游戏规则：
- n行m列棋盘，一个空格(.)，其余格子有X(黑)或O(白)
- 兔兔先手，只能移动与空格相邻的白色棋子(O)移进空格
- 蛋蛋后手，只能移动与空格相邻的黑色棋子(X)移进空格
- 不能操作者输

犯错误定义：兔兔操作前她有必胜策略，操作后蛋蛋有必胜策略。
"""
import sys
sys.setrecursionlimit(100000)


def solve():
    input_data = sys.stdin.read().split()
    idx = 0
    
    n = int(input_data[idx]); idx += 1
    m = int(input_data[idx]); idx += 1
    
    board = []
    for i in range(n):
        row = input_data[idx]; idx += 1
        board.append(row)
    
    k = int(input_data[idx]); idx += 1
    
    # Parse moves: 2k lines, each with x y (1-indexed)
    moves = []
    for i in range(2 * k):
        x = int(input_data[idx]); idx += 1
        y = int(input_data[idx]); idx += 1
        moves.append((x, y))  # 1-indexed
    
    total_cells = n * m
    
    # Find empty cell position from board string
    def find_empty_from_str(board_str):
        for i in range(n):
            for j in range(m):
                if board_str[i*m + j] == '.':
                    return (i, j)
        raise ValueError("No empty cell found")
    
    # Get neighbors of a cell
    def neighbors(r, c):
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < m:
                result.append((nr, nc))
        return result
    
    def execute_move_str(board_str, er, ec, mr, mc):
        """Move piece at (mr,mc) to empty position (er,ec). Returns new board string."""
        chars = list(board_str)
        piece = chars[mr*m + mc]
        chars[er*m + ec] = piece
        chars[mr*m + mc] = '.'
        return ''.join(chars)
    
    # Memoized analysis: returns True if 'player' has a visiting strategy from this state
    state_cache = {}
    
    def analyze_state(board_str, player):
        """Returns True if 'player' has a visiting strategy from this state."""
        key = (board_str, player)
        if key in state_cache:
            return state_cache[key]
        
        # Find empty position
        er, ec = find_empty_from_str(board_str)
        
        # Get possible moves for current player
        possible_moves = []
        for nr, nc in neighbors(er, ec):
            if board_str[nr*m + nc] == player:
                possible_moves.append((nr, nc))
        
        # If no moves, current player loses
        if not possible_moves:
            state_cache[key] = False
            return False
        
        # For each possible move, check if opponent loses from resulting state
        for mr, mc in possible_moves:
            new_board_str = execute_move_str(board_str, er, ec, mr, mc)
            # After current player moves, it's opponent's turn
            if not analyze_state(new_board_str, 'X' if player == 'O' else 'O'):
                state_cache[key] = True
                return True
        
        state_cache[key] = False
        return False
    
    # Build initial board string
    init_board_str = ''.join(board[i][j] for i in range(n) for j in range(m))
    
    # Simulate the game and find rabbit's mistakes
    mistakes = []
    current_board_str = init_board_str
    empty_pos = find_empty_from_str(current_board_str)
    
    for i in range(k):
        # Rabbit's move i (0-indexed)
        rx, ry = moves[2*i]  # rabbit's move
        
        # Before rabbit's move: check if rabbit has visiting strategy
        before_winning = analyze_state(current_board_str, 'O')
        
        # Execute rabbit's move
        r_r, r_c = rx-1, ry-1
        new_board_str = execute_move_str(current_board_str, empty_pos[0], empty_pos[1], r_r, r_c)
        new_empty_pos = (r_r, r_c)  # empty goes to where O was
        
        # After rabbit's move: egg's turn. Check if egg has visiting strategy.
        after_winning = analyze_state(new_board_str, 'X')
        
        # Mistake condition: before rabbit had visiting strategy AND after egg has visiting strategy
        if before_winning and after_winning:
            mistakes.append(i + 1)  # 1-indexed
        
        current_board_str = new_board_str
        empty_pos = new_empty_pos
        
        # Egg's move i
        ex, ey = moves[2*i+1]  # egg's move
        e_r, e_c = ex-1, ey-1
        current_board_str = execute_move_str(current_board_str, empty_pos[0], empty_pos[1], e_r, e_c)
        empty_pos = (e_r, e_c)
    
    print(len(mistakes))
    for mistake in mistakes:
        print(mistake)


if __name__ == "__main__":
    solve()
