import numpy as np
import sys
import os
import glob
import traceback
from IPython.display import clear_output
from itertools import groupby
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

    @staticmethod
    def dump_digraph(boards_record, fname='test', rel_path=None):
        """ dict of md5 -> {board, score, depth, is_max_player, children}"""
        if not rel_path:
            rel_path = './img'
            if not os.path.exists('img'):
                os.mkdir('./img')
            else:
                [os.remove(x) for x in glob.glob('./img/*')]

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

        digraph.render(filename=fname, cleanup=True)


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
        self.player_to_color = {
            ConnectN.HUMAN: 0,
            ConnectN.AI: 1
        }
        self.tt_keys = ConnectN.get_rand_bits(self.sx, self.sy)
        self.zPlayer = list(map(np.uint64, [rd.getrandbits(64),  # HUMAN
                                            rd.getrandbits(64)  # AI
                                    ]))
        self.transposition_table = {}

    # Transposition table storage - node is the lookup key for tt_entry
    #
    # ref: Alpha-Beta with Sibling Prediction Pruning in Chess, Jeroen W.T. Carolus (https://homepages.cwi.nl/~paulk/theses/Carolus.pdf)
    # A typical entry in a transposition table would store the hash key together with the value that comes with
    # the position. This can be an “exact” value  the value of a leaf in the search space, or the value that
    # resulted in a cut-off: an upper bound or a lower bound. Also the depth of the node in the search space
    # must be stored, because a transposition at a depth that is smaller than the current search depth is
    # worthless.
    def update_zobrist_hash(self, zobrist_hash: np.uint64, move_coord: tuple, player: int):
        new_active_player = self.player_to_color[player]
        old_active_player = (new_active_player + 1) % 2
        tt_entry_move_coord = self.tt_keys[(*move_coord, new_active_player)]
        new_zobrist_hash = np.uint64(zobrist_hash) ^ tt_entry_move_coord
        # undo active player and set new
        new_zobrist_hash ^= self.zPlayer[old_active_player]
        new_zobrist_hash ^= self.zPlayer[new_active_player]

        return new_zobrist_hash

    def transposition_table_store(self, hash_key, score, depth, flag):
        self.transposition_table[hash_key] = SimpleNamespace(hash_key=hash_key, score=score, depth=depth, flag=flag)

    @staticmethod
    def get_rand_bits(sx=7, sy=6):
        """ return array of random 64bits with shape (sy, sx, 2) """
        arr = np.array([rd.getrandbits(64) for _ in range(sx * sy * 2)], dtype=np.uint64).reshape(sy, sx, 2)
        assert np.all(arr), arr
        return arr

    @staticmethod
    def get_tt_entry(arr, i, j, sx):
        return arr[j + i * sx]

    def get_new_board(self) -> np.ndarray:
        return np.zeros(shape=(self.sy, self.sx), dtype=int)

    @staticmethod
    def print(c4, use_chars=True, with_row_numbers=True, cinfo=()):

        sy, sx = c4.shape
        if not use_chars:
            symbols = {0: " . ", ConnectN.AI: " o ", ConnectN.HUMAN: " x "}
        else:
            symbols = {0: " . ", ConnectN.AI: " A ", ConnectN.HUMAN: " H "}

        header = [f"{x:02d}." for x in range(sx)]
        if with_row_numbers: header.insert(0, ' ' * 3)
        print(*header)
        for i in range(sy):
            L = [] if not with_row_numbers else [f'{i:02d}.']
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

                    board_data_digraph = {}
                    self.negamax(c4, self.max_depth, -np.inf, +np.inf, ConnectN.AI, solution, board_data_digraph,
                                 zobrist_hash)

                    # solution = self.iterative_deepening_negamax(c4, -np.inf, +np.inf, ConnectN.AI, zobrist_hash)
                    j = solution.get('col')
                    print(
                        f"[AI] {solution} (time:{round(time.process_time() - t0, 3)}s)")

                    # update the topmost board with the best found solution (AI)
                    if self.DUMP_MINIMAX:
                        dump_boards = board_data_digraph.copy()
                        for k, v in board_data_digraph.items():
                            if v['depth'] == self.max_depth:
                                dump_boards[k]['board'], _ = self.play(v['board'], j, True)
                                break
                        print("dumping digraph....")
                        DumpTool.dump_digraph(dump_boards)
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
        def check_diag_win(board, flip=False):
            if flip:
                board = np.flip(board, axis=1)
            for i in range(-sy, sx):
                line = np.diag(board, k=i)
                if not any(line) or len(line) < connect_n:
                    continue
                if has_win(line):
                    return i, 'D0' if not flip else i, 'D1'

        ans = check_diag_win(board)
        if ans:
            return ans
        # anti diagonal
        ans = check_diag_win(board, flip=True)
        if ans:
            return ans

    @staticmethod
    def is_last_move(node):
        return node.flatten().tolist().count(0) == 1

    @staticmethod
    def score(node: np.array, connect_n: int, player) -> float:
        ''' give a high value for a board if maximizer‘s turn or a low value for the board if minimizer‘s turn
            evaluation function: count the number of possible 4 in rows that each player can still make,
            and substract that from each other.
        '''

        sy, sx = node.shape
        score = 0

        assert player in (ConnectN.AI, ConnectN.HUMAN)
        pyr, opp = (ConnectN.AI, ConnectN.HUMAN) if player == ConnectN.AI else (ConnectN.HUMAN, ConnectN.AI)

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

        def get_score_special_horizontal_threat(score):
            """ special threat: horizontal 2-streak with 2 surrounding zeros
                with 1 on the left and 2 on the right or opposite => winning move """
            for i in range(sy):
                line = node[i, :]
                if not any(line):
                    continue

                for j in range(0, len(line) - connect_n):
                    sub_line = line[j:j + connect_n + 1]
                    if i < sy - 1:  # check underneath line is filled
                        sub_line_beneath = node[i + 1, j:j + connect_n + 1]
                        if 0 in sub_line_beneath:
                            # line underneath is not fully filled, no need to check for threat
                            continue

                    if sub_line.tolist() == [0, pyr, pyr, pyr, 0]:
                        return np.inf
                    elif sub_line.tolist() == [0, opp, opp, opp, 0]:
                        return -np.inf

            return score
        score = get_score_special_horizontal_threat(score)
        if score in [np.inf, -np.inf]:
            return score

        # check for a horizontal win
        def check_horiz_win(score):
            for i in range(sy):
                line = node[i, :]
                if not any(line):
                    continue
                for j in range(sx - connect_n + 1):
                    sub_line = line[j:j + connect_n]
                    score = get_updated_score(sub_line, score)
                    if score in [np.inf, -np.inf]:
                        return score
            return score
        score = check_horiz_win(score)
        if score in [np.inf, -np.inf]:
            return score

        # check for a vertical win
        def check_vert_win(score):
            for j in range(sx):
                line = node[:, j]
                if not any(line):
                    continue
                for i in range(0, sy - connect_n + 1):
                    sub_line = line[i:i + connect_n]
                    score = get_updated_score(sub_line, score)
                    if score in [np.inf, -np.inf]:
                        return score
            return score
        score = check_vert_win(score)
        if score in [np.inf, -np.inf]:
            return score

        # check for a diagonal win
        def check_diag_win(score, node):
            for i in range(-sy, sx):
                line = np.diag(node, k=i)
                if not any(line) or len(line) < connect_n:
                    continue
                for j in range(len(line) - connect_n + 1):
                    sub_line = line[j:j + connect_n]
                    score = get_updated_score(sub_line, score)
                    if score in [np.inf, -np.inf]:
                        return score
            return score
        score = check_diag_win(score, node)
        if score in [np.inf, -np.inf]:
            return score

        # anti-diag win
        score = check_diag_win(score, np.flip(node, axis=1))

        return score

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

    @staticmethod
    def quiescent_search(board, alpha, beta, player, connect_n=4):
        """
        Avoid the horizon effect. See https://www.chessprogramming.org/Quiescence_Search
        """

        score = player * ConnectN.score(board, connect_n, ConnectN.AI)

        alpha = max(score, alpha)

        if score >= beta:
            return beta

        # examine every opponent capture and see if this sucks
        sy, sx = board.shape

        for k in range(0, sx):
            zero_indices = np.where(board[:, k] == 0)[0]
            if not len(zero_indices):
                continue

            b, _ = ConnectN.play(board, k, True if player == ConnectN.AI else False)
            score = -ConnectN.quiescent_search(b, -beta, -alpha, -player)

            alpha = max(score, alpha)

            if score >= beta:
                break

        return score

    def negamax(self, board: np.ndarray, depth: int, alpha: float, beta: float, player: int,
                solution: dict, board_data: dict, zobrist_hash: np.uint64 = 0) -> float:

        sy, sx = board.shape

        # tt_entry = self.transposition_table.get(zobrist_hash)
        # if tt_entry and tt_entry.depth >= depth:
        #     # print(f'*** CACHE HIT depth:{depth} alpha:{alpha} beta:{beta} color:{color} tt_entry:{tt_entry} ***')
        #     solution['cache_hit'] = solution.get('cache_hit', 0) + 1
        #     if tt_entry.flag == 'EXACT':  # stored value is exact
        #         return tt_entry.score
        #     elif tt_entry.flag == 'LOWERBOUND':  # update lowerbound beta if needed
        #         alpha = max(alpha, tt_entry.score)
        #     elif tt_entry.flag == 'UPPERBOUND':  # update upperbound beta if needed
        #         beta = min(beta, tt_entry.score)
        #
        #     if alpha >= beta:
        #         return tt_entry.score

        # When the depth limit of the search is exceeded,
        # score the node as if it were a leaf
        if depth == 0 or ConnectN.is_last_move(board):
            score = ConnectN.score(board, self.connect_n, player)
            #    score = player * ConnectN.score(board, self.connect_n, ConnectN.AI)
            #else:
            #    score = ConnectN.quiescent_search(board, alpha, beta, -player)
            # if score <= alpha:
            #     self.transposition_table_store(zobrist_hash, score, depth, 'LOWERBOUND')
            # elif score >= beta:
            #     self.transposition_table_store(zobrist_hash, score, depth, 'UPPERBOUND')
            # else:
            #     self.transposition_table_store(zobrist_hash, score, depth, 'EXACT')
            return score

        if self.DUMP_MINIMAX:
            hsh = hashlib.md5(board.tostring()).hexdigest()
            board_data[hsh] = dict(board=board, depth=depth, hash=hsh,
                                   is_max_player=True if player == ConnectN.AI else False,
                                   children=[])

        score = -np.inf
        for k in range(0, sx):
            zero_indices = np.where(board[:, k] == 0)[0]
            if not len(zero_indices):
                continue

            solution['nb_node_explore'] = solution.get('nb_node_explore', 0) + 1

            play_coord = (zero_indices[-1], k)
            new_zobrist_hash = self.update_zobrist_hash(zobrist_hash, play_coord, player)

            b, _ = ConnectN.play(board, k, True if player == ConnectN.AI else False)
            new_score = -self.negamax(b, depth-1, -beta, -alpha, -player,
                                      solution, board_data, new_zobrist_hash)

            if self.DUMP_MINIMAX:
                hsh_child = hashlib.md5(b.tostring()).hexdigest()
                board_data[hsh]['children'] += [dict(board=b, score=new_score, hash=hsh_child,
                                                     is_max_player=True if -player == ConnectN.AI else False)]
            # else:
            #     print('\t' * depth, f"<{'H' if player == ConnectN.HUMAN else 'AI'}> score:{new_score} depth:{depth} "
            #                         f"played:{play_coord} b:{b[:, k]}", sep='')

            # max(new_score, score)
            if new_score > score:
                score = new_score
                if depth == self.max_depth:
                    assert player == ConnectN.AI, f'(alpha pruning) play:{play_coord}, score:{score} board:\n{board}'
                    solution.update(
                        {'move': play_coord, 'col': k, 'score': score, 'player': 'HUMAN' if player == ConnectN.HUMAN else 'AI'})

            alpha = max(alpha, score)

            # if alpha >= beta:  # or score >= beta ???
            if score >= beta:
                solution['nb_prune'] = solution.get('nb_prune', 0) + 1
                if depth == self.max_depth:
                    assert player == ConnectN.AI,  f'(beta pruning) play:{play_coord}, score:{score} board:\n{board}'
                    solution.update(
                        {'move': play_coord, 'col': k, 'score': score, 'player': 'HUMAN' if player == ConnectN.HUMAN else 'AI'})
                break  # cut-off

        # if score <= alpha: # a lowerbound value
        #     self.transposition_table_store(zobrist_hash, score, depth, 'LOWERBOUND')
        # elif score >= beta: # an upperbound value
        #     self.transposition_table_store(zobrist_hash, score, depth, 'UPPERBOUND')
        # else: # a true minimax value
        #     self.transposition_table_store(zobrist_hash, score, depth, 'EXACT')

        if self.DUMP_MINIMAX:
            board_data[hsh]['score'] = score

        return score


if __name__ == '__main__':
    connect_four = ConnectN(max_depth=4)
    connect_four.DUMP_MINIMAX = False
    connect_four.run()