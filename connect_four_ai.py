import numpy as np
import sys
import os
import traceback
from IPython.display import clear_output
from itertools import groupby
from time import sleep
import random as rd
import time
import hashlib
import graphviz
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from types import SimpleNamespace


class Colors:
    RESET = "\x1b[0m"
    RED = '\033[91m'
    YELLOW = '\033[93m'


class DumpTool:
    colors = ['red', 'white', 'yellow']
    connect_cmap = ListedColormap(colors)

    @staticmethod
    def save_svg(fname: str, format: str, buf: np.ndarray, figsize: tuple = (1.5, 2), default_dir='img'):
        sy, sx = buf.shape

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        # ax = plt.gca()
        ax.set_xlim(auto=True)
        ax.set_ylim(auto=True)
        ax.set_yticks(ticks=[])
        ax.set_xticks(ticks=np.arange(0, sx))
        plt.vlines(x=np.arange(0, sx) + 0.5, ymin=sy -
                                                  1 + 0.5, ymax=-0.5, color="black")
        plt.hlines(y=np.arange(0, sy) + 0.5, xmin=sx -
                                                  1 + 0.5, xmax=-0.5, color="black")
        im = plt.imshow(buf, cmap=DumpTool.connect_cmap)

        fname = os.path.join(os.getcwd(), default_dir,
                             '.'.join([fname, format]))
        plt.savefig(fname, format=format)
        plt.close()  # don't show image


class TTFlags:
    LOWERBOUND = -1
    EXACT = 0
    UPPERBOUND = 1


class ConnectN:
    HUMAN, AI = -1, +1
    DUMP_MINIMAX = False

    def __init__(self, connect_n=4, sx=7, sy=6, max_depth=3):

        assert (
                sx >= connect_n and sy >= connect_n
        ), f"invalid dimensions / n-connect number: ({sx}, {sy}) => ({connect_n}) ?!"

        self.sx, self.sy, self.connect_n, self.max_depth = sx, sy, connect_n, max_depth

        assert max_depth >= 1, 'minimum depth is 1'
        print(f'connect:{self.connect_n} minimax depth:{self.max_depth}')

        # transposition table sx * sy * nb players
        self.tt_keys = ConnectN.get_rand_bits(self.sx, self.sy)
        self.transposition_table = {}


    @staticmethod
    def get_rand_bits(sx=7, sy=6):
        """ return array of random 64bits with shape (sy, sx, 2) """
        arr = np.array([rd.getrandbits(64) for _ in range(sx * sy * 2)], dtype=np.uint64).reshape(sy, sx, 2)
        assert np.all(arr), arr
        return arr

    def get_new_board(self) -> np.ndarray:
        return np.zeros(shape=(self.sy, self.sx), dtype=int)

    @staticmethod
    def dump_digraph(boards_record, fname, rel_path=None):
        if not rel_path:
            rel_path = './img'
        digraph = graphviz.Digraph(format='png')
        for hsh, node in boards_record.items():
            score, depth = node['score'], node['depth']
            max_info = 'MAX' if node['is_max_player'] else 'MIN'
            DumpTool.save_svg(fname=hsh, format='png', buf=node['board'])
            digraph.node(hsh, label=f'{max_info} score={score} depth={depth}',
                         image=f'{rel_path}/{hsh}.png', labelloc='t')

            for c in node['children']:
                child_hash, score, board = c['hash'], c['score'], c['board']
                max_info = 'MAX' if c['is_max_player'] else 'MIN'

                DumpTool.save_svg(fname=child_hash, format='png', buf=board)
                digraph.node(child_hash, f'{max_info} score={score}',
                             image=f'{rel_path}/{child_hash}.png', labelloc='t')
                digraph.edge(hsh, child_hash, constraint='true')

        digraph.render()

    @staticmethod
    def print(c4, cinfo=()):

        sy, sx = c4.shape
        symbols = {0: " . ", 1: " o ", -1: " x "}

        header = [f"{x:02d}." for x in range(sx)]
        print(*header)
        for i in range(sy):
            L = []
            for j in range(sx):
                v = symbols[c4[i, j]]
                L.append(f"{v:3}")

            print(*L, flush=True)

    @staticmethod
    def is_valid_column(board, col):
        _, sx = board.shape
        return 0 <= col < sx and 0 in board[:, col]

    @staticmethod
    def play(board, j, is_max_player):
        sy, sx = board.shape
        if j and not ConnectN.is_valid_column(board, j):
            raise RuntimeError(f"invalid column {j}", file=sys.stderr)

        bc = board.copy()  # don't modify the input

        def drop(j):
            for i in range(sy - 1, -1, -1):
                if bc[i, j] == 0:
                    bc[i, j] = ConnectN.AI if is_max_player else ConnectN.HUMAN
                    return i, j

        if j is not None:
            coord = drop(j)
            assert coord
        else:
            for j in rd.sample(range(sx), sx):
                coord = drop(j)
                if coord:
                    break
        return bc, coord

    def run(self):

        is_max_player = False  # e.g. is AI playing?
        is_game_over = False
        c4 = self.get_new_board()
        in_error = False
        zobrist_hash = 0

        while not in_error:
            try:
                ConnectN.print(c4)
                if is_game_over:
                    break

                clear_output(wait=True)  # (not is_max_player))
                if is_max_player:
                    solution = {}
                    t0 = time.process_time()
                    if False:
                        boards_record = {}
                        score = self.minimax(c4, self.max_depth, alpha=-np.inf, beta=+np.inf, is_max_player=True,
                                             solution=solution, boards_record=boards_record)
                        # assert solution['depth'] == self.max_depth, f'incorrect depth for solution {solution}, max_depth={self.max_depth}'
                    else:
                        # score = self.negamax(c4, self.max_depth, -np.inf, +np.inf, ConnectN.AI, solution, zobrist_hash)
                        solution = self.iterative_deepening_negamax(c4, -np.inf, +np.inf, ConnectN.AI, zobrist_hash)
                    j = solution.get('col')
                    print(
                        f"[AI] {solution} (time:{round(time.process_time() - t0, 3)}s)")

                    # update the topmost board with the best found solution (AI)
                    if self.DUMP_MINIMAX:
                        dump_boards = boards_record.copy()
                        for k, v in boards_record.items():
                            if v['depth'] == self.max_depth:
                                dump_boards[k]['board'] = self.play(
                                    v['board'], j, True)
                                break
                        self.dump_digraph(dump_boards, 'test.png')
                else:
                    j = int(input("play:"))

                c4, coord = ConnectN.play(c4, j, is_max_player)
                # update current cache state
                zobrist_hash = self.update_zobrist_hash(zobrist_hash, coord, ConnectN.HUMAN)

                winfo = ConnectN.is_winning_move(c4, self.connect_n)
                if winfo:
                    is_game_over = True
                    pyr = "AI" if is_max_player else "PLAYER"
                    print(f"*** Game Over! Winner: {pyr} ***")
                    ConnectN.print(c4, winfo)
                    break

                is_max_player = not is_max_player
                in_error = False
            except Exception as e:
                traceback.print_exc()
                in_error = True

    @staticmethod
    def line_count_max_connect(line, val):
        max_connect = 0
        for l in [list(v) for _, v in groupby(line)]:
            if np.equal(l, val).all():
                max_connect = max(max_connect, len(l))
        return max_connect

    @staticmethod
    def is_winning_move(board, connect_n):

        sy, sx = board.shape
        assert (
                connect_n <= sx and connect_n <= sy
        ), f"invalid dimensions / n-connect number: ({sx}, {sy}) => ({connect_n}) ?!"

        def has_win(line):
            for g in groupby(line):
                if g[0] in (ConnectN.AI, ConnectN.HUMAN) and len(list(g[1])) >= connect_n:
                    return True
            return False

        # check for a horizontal win - count consecutive 1s
        for i in range(sy):
            line = board[i, :]
            if not any(line):
                continue
            if has_win(line):
                return (i, 'H')

        # check for a vertical win
        for j in range(sx):
            line = board[:, j]
            if not any(line):
                continue
            if has_win(line):
                return (j, 'V')

        # check for a diagonal win
        for j in range(sx - connect_n + 1):
            for i in range(connect_n - 1, sy):
                line = []
                for k in range(connect_n):
                    line.append(board[i - k, j + k])
                if not any(line):
                    continue
                line = np.array(line)
                if np.all(line == ConnectN.AI) or np.all(line == ConnectN.HUMAN):
                    return ((i, j), 'D0')

        # check for an anti-diagonal win
        for i in range(0, sy - connect_n + 1):
            for j in range(sx - connect_n + 1):
                line = []
                for k in range(connect_n):
                    line.append(board[i + k, j + k])
                line = np.array(line)
                if np.all(line == ConnectN.AI) or np.all(line == ConnectN.HUMAN):
                    return ((i, j), 'D1')

        return None

    @staticmethod
    def is_last_move(node, connect_n):
        return node.reshape(6 * 7).tolist().count(0) == 1

    @staticmethod
    def score(node: np.array, connect_n: int, color) -> float:
        ''' give a high value for a board if maximizer‘s turn or a low value for the board if minimizer‘s turn
            evaluation function: count the number of possible 4 in rows that each player can still make,
            and substract that from each other.
        '''

        sy, sx = node.shape
        score = 0

        assert color in (ConnectN.AI, ConnectN.HUMAN)
        pyr, opp = (ConnectN.AI, ConnectN.HUMAN) if color == ConnectN.AI else (ConnectN.AI, ConnectN.HUMAN)

        def check_feature_2a(node, line, line_under=None):
            """ feature 2.a: A move can be made on either immediately adjacent columns """
            nb_connect = line.tolist().count(pyr)
            nb_move = 0
            if nb_connect == 2:
                if line[0] == 0 and (not line_under or line_under[0] != 0):
                    nb_move += 1
                if line[-1] == 0 and (not line_under or line_under[-1] != 0):
                    nb_move += 1
            return score

        def get_updated_score(line, score):
            """ Compute score based on heuristic function """
            assert len(line) == connect_n, f'invalid line {line}'

            nb_connect_pyr = line.tolist().count(pyr)
            nb_connect_opp = line.tolist().count(opp)
            nb_connect_empty = line.tolist().count(0)
            # feature 1: chessmen are connected
            if nb_connect_pyr == 4:
                return +np.inf
            elif nb_connect_pyr == 3 and nb_connect_empty == 1:
                score += 300
            elif nb_connect_pyr == 2 and nb_connect_empty == 2:
                score += 200

            if nb_connect_opp == 3 and nb_connect_empty == 1:
                score -= 500
            elif nb_connect_opp == 4:
                return -np.inf

            # https://pdfs.semanticscholar.org/f323/3fa36a5026b42c7f331a5c98e66aad9d3e8c.pdf
            return score

        # special threat: horizontal 2-streak with 2 surrounding zeros
        # with 1 on the left and 2 on the right or opposite => winning move
        for i in range(sy):
            line = node[i, :]

            if not any(line):
                continue
            elif i < sy - 1 and 0 in node[i + 1, :]:
                # line underneath is not fully filled, no need to check for threat
                continue

            for i in range(0, len(line) - connect_n + 2):
                sub_line = line[i:i + connect_n + 1]
                # check underneath line is filled
                nb_connect_pyr = sub_line.tolist().count(pyr)
                nb_connect_opp = sub_line.tolist().count(opp)
                nb_connect_empty = sub_line.tolist().count(0)
                if nb_connect_pyr == 2 and nb_connect_empty == 3 and \
                        sub_line[0] == 0 and sub_line[-1] == 0:
                    # print('plyr winner move')
                    score += np.inf
                elif nb_connect_opp == 2 and nb_connect_empty == 3 and \
                        sub_line[0] == 0 and sub_line[-1] == 0:
                    # print('opp winner move')
                    score -= np.inf

        # check for a horizontal win
        for i in range(sy):
            line = node[i, :]
            if not any(line):
                continue
            for j in range(sx - connect_n + 1):
                sub_line = line[j:j + connect_n]
                score = get_updated_score(sub_line, score)

        # check for a vertical win
        for j in range(sx):
            line = node[:, j]
            if not any(line):
                continue
            for i in range(0, sy - connect_n + 1):
                sub_line = line[i:i + connect_n]
                score = get_updated_score(sub_line, score)

        # check for a diagonal win
        for j in range(sx - connect_n + 1):
            for i in range(connect_n - 1, sy):
                sub_line = []
                for k in range(connect_n):
                    sub_line.append(node[i - k, j + k])
                if not any(line):
                    continue
                score = get_updated_score(np.array(sub_line), score)

        # check for an anti-diagonal win
        for i in range(0, sy - connect_n + 1):
            for j in range(sx - connect_n + 1):
                sub_line = []
                for k in range(connect_n):
                    sub_line.append(node[i + k, j + k])
                score = get_updated_score(np.array(sub_line), score)

        return score

    def minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, is_max_player: bool, solution,
                boards_record: dict) -> float:

        # print('depth=', depth)
        assert alpha < beta

        sy, sx = board.shape

        # When the depth limit of the search is exceeded,
        # score the node as if it were a leaf
        # The heuristic value is a score measuring the favorability of the node for the maximizing player.
        if depth == 0 or ConnectN.is_last_move(board, self.connect_n) or ConnectN.is_winning_move(board,
                                                                                                  self.connect_n):
            return ConnectN.score(board, self.connect_n)

        if self.DUMP_MINIMAX:
            root_board_hash = self.hash_board(board)
            boards_record[root_board_hash] = {'board': board, 'children': [
            ], 'score': None, 'depth': depth, 'is_max_player': is_max_player}

        if is_max_player:
            value = -np.inf
            for k in range(0, sx):
                if 0 in board[:, k]:
                    solution['nb_node_explore'] = solution.get(
                        'nb_node_explore', 0) + 1
                    b = ConnectN.play(board, k, True)

                    value_new = self.minimax(
                        b, depth - 1, alpha, beta, (not is_max_player), solution, boards_record)

                    if self.DUMP_MINIMAX:
                        boards_record[root_board_hash]['children'].append({
                            'board': b,
                            'hash': self.hash_board(b),
                            'score': value_new,
                            'is_max_player': is_max_player if depth > 1 else not is_max_player
                        })

                    if value_new > value:  # maximize value
                        value = value_new
                        if depth == self.max_depth:
                            solution.update(
                                {'col': k, 'depth': depth, 'score': value, 'is_max_player': is_max_player})

                    if value >= beta:  # beta pruning
                        solution['nb_beta_prune'] = solution.get(
                            'nb_beta_prune', 0) + 1
                        break

                    alpha = max(alpha, value)  # no fail-soft

        else:
            value = +np.inf
            for k in range(0, sx):
                if 0 in board[:, k]:
                    solution['nb_node_explore'] = solution.get(
                        'nb_node_explore', 0) + 1
                    b = ConnectN.play(board, k, False)

                    value_new = self.minimax(
                        b, depth - 1, alpha, beta, (not is_max_player), solution, boards_record)

                    if self.DUMP_MINIMAX:
                        boards_record[root_board_hash]['children'].append({
                            'board': b,
                            'hash': self.hash_board(b),
                            'score': value_new,
                            'is_max_player': is_max_player if depth > 1 else not is_max_player
                        })

                    if value_new < value:  # minimize value
                        value = value_new
                        # solution.update({'col': k, 'depth': depth, 'score': value,'is_max_player': is_max_player})

                    if value <= alpha:  # alpha pruning
                        solution['nb_alpha_prune'] = solution.get(
                            'nb_alpha_prune', 0) + 1
                        break

                    beta = min(beta, value)  # no fail-soft

        if self.DUMP_MINIMAX:
            boards_record[root_board_hash]['score'] = value
        return value

    # Transposition table storage - node is the lookup key for tt_entry
    #
    # ref: Alpha-Beta with Sibling Prediction Pruning in Chess, Jeroen W.T. Carolus (https://homepages.cwi.nl/~paulk/theses/Carolus.pdf)
    # A typical entry in a transposition table would store the hash key together with the value that comes with
    # the position. This can be an “exact” value  the value of a leaf in the search space, or the value that
    # resulted in a cut-off: an upper bound or a lower bound. Also the depth of the node in the search space
    # must be stored, because a transposition at a depth that is smaller than the current search depth is
    # worthless.
    def update_zobrist_hash(self, zobrist_hash, move_coord, color):
        tt_entry_move_coord = self.tt_keys[(*move_coord, 0 if color == ConnectN.HUMAN else 1)]
        return np.uint64(zobrist_hash) ^ tt_entry_move_coord

    def transposition_table_store(self, hash_key, score, depth, flag):
        self.transposition_table[hash_key] = SimpleNamespace(hash_key=hash_key, score=score, depth=depth, flag=flag)

    def iterative_deepening_negamax(self, board: np.ndarray, alpha: float, beta: float, color: int,
                                    zobrist_hash: np.uint64 = 0, max_sec_elapsed: int = 2) -> float:

        solution_iterative_deepening = {}

        depth = 1
        self.max_depth = depth
        self.transposition_table.clear()

        start_time = time.process_time()
        while time.process_time() - start_time <= max_sec_elapsed:
            print(f'iterative deepening, processing for max depth: {depth}')
            solution = {}
            self.negamax(board, depth, alpha, beta, color, solution, zobrist_hash)
            print('solution:', solution)
            solution_iterative_deepening = solution.copy()

            depth += 1
            self.max_depth = depth

        return solution_iterative_deepening

    def negamax(self, board: np.ndarray, depth: int, alpha: float, beta: float, color: int,
                solution: dict, zobrist_hash: np.uint64 = 0) -> float:

        sy, sx = board.shape

        tt_entry = self.transposition_table.get(zobrist_hash)
        if tt_entry and tt_entry.depth >= depth:
            # print(f'*** CACHE HIT depth:{depth} alpha:{alpha} beta:{beta} color:{color} tt_entry:{tt_entry} ***')
            solution['cache_hit'] = solution.get('cache_hit', 0) + 1
            if tt_entry.flag == 'EXACT': # stored value is exact
                return tt_entry.score
            elif tt_entry.flag == 'LOWERBOUND': # update lowerbound beta if needed
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == 'UPPERBOUND': # update upperbound beta if needed
                beta = min(beta, tt_entry.score)

            if alpha >= beta:
                return tt_entry.score

        # When the depth limit of the search is exceeded,
        # score the node as if it were a leaf
        if depth == 0 or ConnectN.is_last_move(board, self.connect_n) or ConnectN.is_winning_move(board,
                                                                                                  self.connect_n):
            score = color * ConnectN.score(board, self.connect_n, ConnectN.AI)
            if score <= alpha:
                self.transposition_table_store(zobrist_hash, score, depth, 'LOWERBOUND')
            elif score >= beta:
                self.transposition_table_store(zobrist_hash, score, depth, 'UPPERBOUND')
            else:
                self.transposition_table_store(zobrist_hash, score, depth, 'EXACT')
            return score

        score = -np.inf
        for k in range(0, sx):
            zero_indices = np.where(board[:, k] == 0)[0]
            if len(zero_indices):
                solution['nb_node_explore'] = solution.get(
                    'nb_node_explore', 0) + 1

                move_coord = (zero_indices[-1], k)
                new_zobrist_hash = self.update_zobrist_hash(zobrist_hash, move_coord, color)

                b, _ = ConnectN.play(board, k, True if color == ConnectN.AI else False)
                new_score = -self.negamax(b, depth-1, -beta, -alpha, -color,
                                          solution, new_zobrist_hash)

                if new_score > score:
                    score = new_score
                    if depth == self.max_depth:
                        solution.update(
                            {'col': k, 'score': score, 'color': color})

                alpha = max(alpha, score)

                # if alpha >= beta:  # or score >= beta ???
                if score >= beta:
                    solution['nb_prune'] = solution.get('nb_prune', 0) + 1
                    break  # cut-off

        if score <= alpha: # a lowerbound value
            self.transposition_table_store(zobrist_hash, score, depth, 'LOWERBOUND')
        elif score >= beta: # an upperbound value
            self.transposition_table_store(zobrist_hash, score, depth, 'UPPERBOUND')
        else: # a true minimax value
            self.transposition_table_store(zobrist_hash, score, depth, 'EXACT')

        return score


if __name__ == '__main__':
    connect_four = ConnectN(max_depth=6)
    connect_four.DUMP_MINIMAX = False
    connect_four.run()