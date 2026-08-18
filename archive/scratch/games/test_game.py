"""
暴力测试：随机生成小棋盘游戏，验证结果
"""
import random
import subprocess
import sys


def generate_random_game(n, m, k):
    """Generate a random valid game."""
    # Place one empty cell, rest are X or O
    total = n * m
    empty_pos = random.randint(0, total - 1)
    
    board_chars = []
    for i in range(total):
        if i == empty_pos:
            board_chars.append('.')
        else:
            board_chars.append(random.choice('XO'))
    
    # Format into rows
    board_rows = []
    for i in range(n):
        row = ''.join(board_chars[i*m:(i+1)*m])
        board_rows.append(row)
    
    # Generate valid moves
    moves = []
    empty_r, empty_c = empty_pos // m, empty_pos % m
    
    def get_neighbors(r, c):
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < n and 0 <= nc < m:
                result.append((nr, nc))
        return result
    
    def do_move(board_str, er, ec, mr, mc):
        chars = list(board_str)
        piece = chars[mr*m + mc]
        chars[er*m + ec] = piece
        chars[mr*m + mc] = '.'
        return ''.join(chars), (mr, mc)
    
    board_str = ''.join(board_chars)
    
    for i in range(k):
        # Rabbit's move: must be adjacent O
        possible = []
        for nr, nc in get_neighbors(empty_r, empty_c):
            if board_str[nr*m + nc] == 'O':
                possible.append((nr, nc))
        
        if not possible:
            break  # Game would end
        
        mr, mc = random.choice(possible)
        moves.append((mr+1, mc+1))  # 1-indexed
        board_str, (empty_r, empty_c) = do_move(board_str, empty_r, empty_c, mr, mc)
        
        # Egg's move: must be adjacent X
        possible = []
        for nr, nc in get_neighbors(empty_r, empty_c):
            if board_str[nr*m + nc] == 'X':
                possible.append((nr, nc))
        
        if not possible:
            break  # Game would end
        
        mr, mc = random.choice(possible)
        moves.append((mr+1, mc+1))  # 1-indexed
        board_str, (empty_r, empty_c) = do_move(board_str, empty_r, empty_c, mr, mc)
    
    if len(moves) < 2*k:
        return None
    
    # Build input
    lines = [f"{n} {m}"]
    for row in board_rows:
        lines.append(row)
    lines.append(str(k))
    for move in moves:
        lines.append(f"{move[0]} {move[1]}")
    
    return '\n'.join(lines)


def run_solution(input_str):
    result = subprocess.run(
        [sys.executable, "solve_game.py"],
        input=input_str,
        capture_output=True,
        text=True,
        timeout=30
    )
    return result.stdout.strip()


def main():
    random.seed(42)
    
    for test in range(50):
        n = random.randint(1, 4)
        m = random.randint(1, 4)
        k = random.randint(1, 3)
        
        game = generate_random_game(n, m, k)
        if game is None:
            continue
        
        expected_output = run_solution(game)
        print(f"Test {test+1}: n={n}, m={m}, k={k}")
        print(f"  Input:\n{game}")
        print(f"  Output: {expected_output}")
        print()


if __name__ == "__main__":
    main()
