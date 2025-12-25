#!/usr/bin/env python3

import chess
import sys
import time
import threading
from collections import defaultdict, namedtuple

INF = 99999999

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Небольшие PST для примера
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ]
}

TTEntry = namedtuple("TTEntry", ["depth", "flag", "score", "best_move"])
# flag: 'EXACT', 'LOWER', 'UPPER'

# ---- Утилиты ----

def fast_board_key(board: chess.Board):
    return (board.board_fen(), board.turn, board.castling_xfen(), board.ep_square, board.halfmove_clock)


def mvv_lva_score(board, move):
    """
    MVV-LVA с учётом en-passant (если применимо) и премией за промоцию.
    Чем выше — тем раньше ход.
    """
    score = 0
    if board.is_capture(move):
        # определить жертву: для en-passant жертва — пешка не на to_square
        if board.is_en_passant(move):
            victim_value = PIECE_VALUES[chess.PAWN]
        else:
            victim = board.piece_at(move.to_square)
            victim_value = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        score += victim_value * 10 - attacker_value
    if move.promotion:
        # повышение предпочтения для промоции
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score

# ---- Оценка позиции ----

def evaluate(board: chess.Board):

    if board.is_checkmate():
        return -INF + 1
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    material = 0
    pst_score = 0

    for piece_type in PIECE_VALUES:
        for sq in board.pieces(piece_type, chess.WHITE):
            material += PIECE_VALUES[piece_type]
            if piece_type in PST:
                pst_score += PST[piece_type][sq]
        for sq in board.pieces(piece_type, chess.BLACK):
            material -= PIECE_VALUES[piece_type]
            if piece_type in PST:
                pst_score -= PST[piece_type][chess.square_mirror(sq)]

    # mobility: безопасный подсчёт
    mobility_count = sum(1 for _ in board.legal_moves)
    mobility = 10 * mobility_count

    check_bonus = -50 if board.is_check() else 0

    # ========= BISHOP =========
    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        blocked = 0
        for pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
            if chess.square_color(sq) == chess.square_color(pawn_sq):
                blocked += 1
        score -= blocked * 4

    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        blocked = 0
        for pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
            if chess.square_color(sq) == chess.square_color(pawn_sq):
                blocked += 1
        score += blocked * 4


    DEV_SQUARES_WHITE = {chess.C4, chess.B5, chess.E3, chess.F4, chess.G5}
    DEV_SQUARES_BLACK = {chess.C5, chess.B4, chess.E6, chess.F5, chess.G4}

    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        if sq in DEV_SQUARES_WHITE:
            score += 15

    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        if sq in DEV_SQUARES_BLACK:
            score -= 15

    if board.fullmove_number < 8:
        for sq in board.pieces(chess.QUEEN, chess.WHITE):
            score -= 20
        for sq in board.pieces(chess.QUEEN, chess.BLACK):
            score += 20


    score_white = material + pst_score + mobility + check_bonus
    return score_white if board.turn == chess.WHITE else -score_white

# ---- TT и state ----
class SearchState:
    def __init__(self):
        self.tt = {}
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.history = defaultdict(int)

class SearchAbort(Exception):
    pass


def quiescence(board: chess.Board, alpha: int, beta: int, state: SearchState, stop_event: threading.Event):
    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return alpha
    captures.sort(key=lambda mv: -mvv_lva_score(board, mv))

    for move in captures:
        if stop_event.is_set():
            raise SearchAbort()
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, state, stop_event)
        finally:
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

# ---- Negamax с alpha-beta и TT ----

def negamax(board: chess.Board, depth: int, alpha: int, beta: int, state: SearchState, stop_event: threading.Event):
    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1

    if depth == 0:
        return quiescence(board, alpha, beta, state, stop_event)

    key = fast_board_key(board)
    tt_entry = state.tt.get(key)
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == 'EXACT':
            return tt_entry.score
        elif tt_entry.flag == 'LOWER':
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == 'UPPER':
            beta = min(beta, tt_entry.score)
        if alpha >= beta:
            return tt_entry.score

    alpha_orig = alpha
    beta_orig = beta

    best_score = -INF
    best_move = None

    moves = list(board.legal_moves)

    def move_key(mv):
        if tt_entry and tt_entry.best_move and mv == tt_entry.best_move:
            return (0, 0, 0)
        cap = 0 if board.is_capture(mv) else 1
        mvv = -mvv_lva_score(board, mv)
        hist = -state.history[(board.turn, mv.from_square, mv.to_square)]
        return (cap, mvv, hist)

    moves.sort(key=move_key)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()
        mover = board.turn  # сторона, делающая ход
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, state, stop_event)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            # обновляем history для хода, который привёл к улучшению
            if not board.is_capture(move):
                state.history[(mover, move.from_square, move.to_square)] += 2 ** depth

        if alpha >= beta:
            # beta-cutoff: запомним ход
            state.history[(mover, move.from_square, move.to_square)] += 2 ** depth
            break

    if best_score >= beta_orig:
        flag = 'LOWER'
    elif best_score <= alpha_orig:
        flag = 'UPPER'
    else:
        flag = 'EXACT'

    state.tt[key] = TTEntry(depth=depth, flag=flag, score=best_score, best_move=best_move)
    return best_score

# ---- SearchThread (итеративное углубление) ----
class SearchThread(threading.Thread):
    def __init__(self, root_board: chess.Board, wtime=None, btime=None, winc=0, binc=0, movetime=None, max_depth=None, stop_event=None):
        super().__init__()
        self.root_board = root_board.copy()
        self.wtime = wtime
        self.btime = btime
        self.winc = winc or 0
        self.binc = binc or 0
        self.movetime = movetime
        self.max_depth = max_depth
        self.stop_event = stop_event or threading.Event()

        self.best_move = None
        self.best_score = None
        self.depth_reached = 0

        self.state = SearchState()
        self.state.time_limit = 0.0
        self.state.start_time = 0.0

    def time_remaining_ms(self):
        if self.movetime:
            return self.movetime

        # определяем оставшееся время
        if self.root_board.turn == chess.WHITE:
            remaining = self.wtime
            inc = self.winc
        else:
            remaining = self.btime
            inc = self.binc

        if remaining is None:
            return 10000

        # ---- PANIC MODE (флаг падает) ----
        # при <= 1 секунде поиск не имеет смысла
        if remaining <= 1000:
            return 0

        # ---- LOW TIME MODE ----
        # быстрый, неглубокий поиск
        if remaining <= 3000:
            return max(20, remaining // 40)


        return max(20, remaining // 20 + inc * 2)

    def run(self):
        ms = self.time_remaining_ms()
        self.state.time_limit = ms / 1000.0
        self.state.start_time = time.time()

        depth = 1
        try:
            while not self.stop_event.is_set():
                if self.max_depth and depth > self.max_depth:
                    break
                self.depth_reached = depth

                # корневое упорядочивание
                moves = list(self.root_board.legal_moves)
                root_key = fast_board_key(self.root_board)
                root_tt = self.state.tt.get(root_key)

                def root_key_fn(mv):
                    if root_tt and root_tt.best_move and mv == root_tt.best_move:
                        return (0, 0)
                    cap = 0 if self.root_board.is_capture(mv) else 1
                    mvv = -mvv_lva_score(self.root_board, mv)
                    return (cap, mvv)

                moves.sort(key=root_key_fn)

                best_for_depth = None
                best_score_for_depth = -INF

                for mv in moves:
                    if self.stop_event.is_set():
                        break
                    mover = self.root_board.turn
                    # push once, pop once (без двойного pop даже при исключениях)
                    self.root_board.push(mv)
                    try:
                        score = -negamax(self.root_board, depth - 1, -INF, INF, self.state, self.stop_event)
                    except SearchAbort:
                        # просто пробрасываем, но НЕ вызываем pop здесь
                        raise
                    finally:
                        # гарантированно снимаем ход ровно один раз
                        self.root_board.pop()

                    if score > best_score_for_depth:
                        best_score_for_depth = score
                        best_for_depth = mv

                    # тайм-чек между корневыми ходами
                    if (time.time() - self.state.start_time) > self.state.time_limit:
                        break

                if best_for_depth is not None:
                    self.best_move = best_for_depth
                    self.best_score = best_score_for_depth
                    elapsed = time.time() - self.state.start_time
                    nps = int(self.state.nodes / elapsed) if elapsed > 0 else 0
                    try:
                        pv_str = self.best_move.uci()
                    except Exception:
                        pv_str = "-"
                    print(f"info depth {depth} score cp {best_score_for_depth} time {int(elapsed*1000)} nodes {self.state.nodes} nps {nps} pv {pv_str}")
                    sys.stdout.flush()

                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break

                depth += 1

        except SearchAbort:
            # корректное окончание
            pass
        except Exception as e:
            print("Search error:", e, file=sys.stderr)
            sys.stderr.flush()

        # По завершении — печатаем bestmove (UCI требует вывод bestmove при завершении поиска)
        if self.best_move:
            try:
                print(f"bestmove {self.best_move.uci()}")
                sys.stdout.flush()
            except Exception:
                print("bestmove 0000")
                sys.stdout.flush()
        else:
            # если ничего не найдено, попробуем любой легальный ход
            try:
                fb = next(iter(self.root_board.legal_moves))
                print(f"bestmove {fb.uci()}")
                sys.stdout.flush()
            except StopIteration:
                print("bestmove 0000")
                sys.stdout.flush()

# ---- UCI loop ----

def uci_loop():
    board = chess.Board()
    search_thread = None
    stop_event = threading.Event()
    print("id name DarkOnEngine")
    print("id author Dark and Classic")
    print("uciok")
    sys.stdout.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "":
                continue
            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                print("id name DarkOnEngine")
                print("id author Dark and Classic")
                print("uciok")
                sys.stdout.flush()
            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()
            elif cmd == "ucinewgame":
                board = chess.Board()
            elif cmd == "position":
                idx = 1
                if len(parts) >= 2 and parts[1] == "startpos":
                    board = chess.Board()
                    idx = 2
                elif len(parts) >= 2 and parts[1] == "fen":
                    if len(parts) >= 8:
                        fen = " ".join(parts[2:8])
                        try:
                            board = chess.Board(fen)
                        except Exception:
                            board = chess.Board()
                        idx = 8
                if idx < len(parts) and parts[idx] == "moves":
                    for mv in parts[idx+1:]:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            pass
            elif cmd == "go":
                wtime = btime = winc = binc = movetime = None
                depth = None
                i = 1
                while i < len(parts):
                    if parts[i] == "wtime":
                        try:
                            wtime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "btime":
                        try:
                            btime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "winc":
                        try:
                            winc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "binc":
                        try:
                            binc = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "movetime":
                        try:
                            movetime = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    elif parts[i] == "depth":
                        try:
                            depth = int(parts[i+1]); i += 2
                        except Exception:
                            i += 1
                    else:
                        i += 1

                # остановим предыдущий поиск, если есть
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                    stop_event.clear()

                stop_event = threading.Event()
                search_thread = SearchThread(board, wtime=wtime, btime=btime, winc=winc or 0, binc=binc or 0, movetime=movetime, max_depth=depth, stop_event=stop_event)
                # запускаем асинхронно — поток сам выведет bestmove при завершении
                search_thread.start()


            elif cmd == "stop":
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=2.0)
                # Ничего дополнительно не печатаем — поток печатает bestmove при завершении

            elif cmd == "quit":
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=2.0)
                break
            else:
                pass

        except Exception as e:
            print("error:", e, file=sys.stderr)
            sys.stderr.flush()
            break

if __name__ == "__main__":
    uci_loop()
