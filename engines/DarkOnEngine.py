#!/usr/bin/env python3
"""
DarkOnEngine (menschliche Version, Ziel ca. 1700 Elo)

Ideen:
  * Suche nur flach (2-4 Halbzuege + Quiescence + Schachverlaengerung)
  * Zugauswahl mit "menschlichem" Rauschen und gelegentlichen Fehlern
    (mehr Fehler bei Zeitnot)
  * Bedenkzeit wie ein Mensch: schnell in der Eroeffnung, bei Schlagzuegen
    und Einzelzuegen, laenger in komplexen Stellungen, gelegentlich lange
    Denkpausen. Die Engine wartet bewusst, auch wenn sie schon fertig ist.
  * Kleines Eroeffnungsbuch
"""
import sys
import time
import math
import threading
import random as rnd
from collections import namedtuple

import chess

INF = 10 ** 9
MATE = 100_000
MAXPLY = 40

# ── Staerke-Parameter (hier ggf. nachjustieren) ───────────────────────────────
ROOT_MARGIN = 180      # cp: nur Zuege innerhalb dieser Spanne werden bewertet
NOISE_SIGMA = 32       # cp: Rauschen auf Zugbewertungen
NOISE_SIGMA_PANIC = 65 # cp: Rauschen bei Zeitnot
BLUNDER_P = 0.025      # Chance auf einen "Aussetzer"
BLUNDER_P_PANIC = 0.07

VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000}

# ── Positionstabellen (a1 = Index 0, Weiss-Sicht, Reihe 1 zuerst) ─────────────
PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  5, 20, 20,  5,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
}
# FIX: Die Koenigstabelle war im Original auf den Kopf gestellt
# (der Koenig wurde in Richtung 8. Reihe gelockt). Hier korrekt: Reihe 1 zuerst.
KING_MID = [
     20, 30, 10,  0,  0, 10, 30, 20,
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
]
KING_END = [
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-10,  5,  5,  5,  5,-10,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,-10,  5,  5,  5,  5,-10,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]
PHASE_W = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PASSED = [0, 10, 20, 35, 60, 100, 140, 0]


# ══════════════════════════════════════════════════════════════════════════════
#  BEWERTUNG (aus Sicht des Spielers am Zug)
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(board: chess.Board) -> int:
    score = 0
    phase = 0
    pawns = {True: [], False: []}
    pfiles = {True: [0] * 8, False: [0] * 8}
    bishops = {True: 0, False: 0}
    rooks = {True: [], False: []}
    kings = {}
    minors_home = {True: 0, False: 0}

    for sq, p in board.piece_map().items():
        pt, col = p.piece_type, p.color
        s = 1 if col else -1
        idx = sq if col else sq ^ 56
        if pt == chess.KING:
            kings[col] = sq
            continue
        score += s * VAL[pt]
        if pt in PST:
            score += s * PST[pt][idx]
        phase += PHASE_W.get(pt, 0)
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if pt == chess.PAWN:
            pawns[col].append((f, r))
            pfiles[col][f] += 1
        elif pt == chess.BISHOP:
            bishops[col] += 1
            if r == (0 if col else 7):
                minors_home[col] += 1
        elif pt == chess.KNIGHT:
            if r == (0 if col else 7):
                minors_home[col] += 1
        elif pt == chess.ROOK:
            rooks[col].append((f, r))

    phase = min(phase, 24)

    # Koenig (Mittelspiel <-> Endspiel interpoliert)
    for col, ksq in kings.items():
        s = 1 if col else -1
        idx = ksq if col else ksq ^ 56
        score += s * ((KING_MID[idx] * phase + KING_END[idx] * (24 - phase)) // 24)
        # Bauernschutz im Mittelspiel
        if phase > 12:
            kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
            d = 1 if col else -1
            shield = 0
            for (pf, pr) in pawns[col]:
                if abs(pf - kf) <= 1:
                    if pr == kr + d:
                        shield += 10
                    elif pr == kr + 2 * d:
                        shield += 5
            score += s * shield
            if ksq in ((chess.G1, chess.C1, chess.B1) if col
                       else (chess.G8, chess.C8, chess.B8)):
                score += s * 25  # Rochade-aehnliche Koenigsposition

    for col in (True, False):
        s = 1 if col else -1
        enemy = pawns[not col]
        # Laeuferpaar
        if bishops[col] >= 2:
            score += s * 30
        # Bauernstruktur
        for f in range(8):
            n = pfiles[col][f]
            if n > 1:
                score -= s * 12 * (n - 1)
            if n and (f == 0 or not pfiles[col][f - 1]) and (f == 7 or not pfiles[col][f + 1]):
                score -= s * 10
        for (f, r) in pawns[col]:
            blocked = False
            for (ef, er) in enemy:
                if abs(ef - f) <= 1 and ((er > r) if col else (er < r)):
                    blocked = True
                    break
            if not blocked:
                rr = r if col else 7 - r
                score += s * (PASSED[rr] * (24 + (24 - phase)) // 36)
        # Tuerme
        for (f, r) in rooks[col]:
            if not pfiles[col][f]:
                score += s * (20 if not pfiles[not col][f] else 10)
            if r == (6 if col else 1):
                score += s * 15
        # Entwicklung in der Eroeffnung
        if board.fullmove_number <= 10:
            score -= s * 10 * minors_home[col]

    score += 10  # Tempo (Weiss)
    return score if board.turn else -score


# ══════════════════════════════════════════════════════════════════════════════
#  SUCHE
# ══════════════════════════════════════════════════════════════════════════════
TTEntry = namedtuple("TTEntry", "depth flag score move")


class SearchAbort(Exception):
    pass


class SearchState:
    def __init__(self, stop_event, deadline):
        self.tt = {}
        self.history = {}
        self.killers = [[None, None] for _ in range(MAXPLY + 8)]
        self.nodes = 0
        self.stop_event = stop_event
        self.deadline = deadline
        self.no_abort = True   # Tiefe 1 darf nie abgebrochen werden

    def tick(self):
        self.nodes += 1
        if self.no_abort:
            return
        if (self.nodes & 127) == 0:
            if self.stop_event.is_set() or time.time() > self.deadline:
                raise SearchAbort()


def tt_key(board):
    return board._transposition_key()


def order_moves(board, moves, st, ply, tt_move):
    kl = st.killers[min(ply, MAXPLY + 7)]
    turn = board.turn

    def score(m):
        if tt_move is not None and m == tt_move:
            return 10 ** 7
        if board.is_capture(m):
            if board.is_en_passant(m):
                victim = 100
            else:
                victim = VAL.get(board.piece_type_at(m.to_square), 0)
            att = VAL.get(board.piece_type_at(m.from_square), 0)
            return 10 ** 6 + victim * 10 - att
        if m.promotion:
            return 9 * 10 ** 5
        if m == kl[0] or m == kl[1]:
            return 8 * 10 ** 5
        return st.history.get((turn, m.from_square, m.to_square), 0)

    moves.sort(key=score, reverse=True)
    return moves


def quiescence(board, alpha, beta, ply, qd, st):
    st.tick()
    stand = evaluate(board)
    if stand >= beta:
        return stand
    if qd >= 6:
        return stand
    if stand > alpha:
        alpha = stand
    caps = list(board.generate_legal_captures())
    caps = order_moves(board, caps, st, ply, None)
    for m in caps:
        if not m.promotion:
            if board.is_en_passant(m):
                victim = 100
            else:
                victim = VAL.get(board.piece_type_at(m.to_square), 0)
            if stand + victim + 200 < alpha:   # Delta-Pruning
                continue
        board.push(m)
        sc = -quiescence(board, -beta, -alpha, ply + 1, qd + 1, st)
        board.pop()
        if sc >= beta:
            return sc
        if sc > alpha:
            alpha = sc
    return alpha


def negamax(board, depth, alpha, beta, ply, st):
    st.tick()

    if ply > 0 and (board.is_repetition(2) or board.halfmove_clock >= 100
                    or board.is_insufficient_material()):
        return 0

    in_check = board.is_check()
    if in_check and ply < MAXPLY:
        depth += 1
    if depth <= 0:
        return quiescence(board, alpha, beta, ply, 0, st)

    key = tt_key(board)
    e = st.tt.get(key)
    tt_move = None
    if e is not None:
        tt_move = e.move
        if e.depth >= depth and ply > 0:
            if e.flag == 'EXACT':
                return e.score
            if e.flag == 'LOWER' and e.score >= beta:
                return e.score
            if e.flag == 'UPPER' and e.score <= alpha:
                return e.score

    moves = list(board.legal_moves)
    if not moves:
        return -(MATE - ply) if in_check else 0
    moves = order_moves(board, moves, st, ply, tt_move)

    best = -INF
    best_move = None
    a0 = alpha
    for i, m in enumerate(moves):
        quiet = not board.is_capture(m) and not m.promotion
        board.push(m)
        if i >= 4 and depth >= 3 and quiet and not in_check:
            sc = -negamax(board, depth - 2, -alpha - 1, -alpha, ply + 1, st)
            if sc > alpha:
                sc = -negamax(board, depth - 1, -beta, -alpha, ply + 1, st)
        else:
            sc = -negamax(board, depth - 1, -beta, -alpha, ply + 1, st)
        board.pop()

        if sc > best:
            best, best_move = sc, m
        if sc > alpha:
            alpha = sc
        if alpha >= beta:
            if quiet:
                k = st.killers[min(ply, MAXPLY + 7)]
                if k[0] != m:
                    k[1], k[0] = k[0], m
                hk = (board.turn, m.from_square, m.to_square)
                st.history[hk] = st.history.get(hk, 0) + depth * depth
            break

    flag = 'LOWER' if best >= beta else ('UPPER' if best <= a0 else 'EXACT')
    if len(st.tt) > 400_000:
        st.tt.clear()
    st.tt[key] = TTEntry(depth, flag, best, best_move)
    return best


def search_root(board, depth, st, prev):
    """Bewertet alle Wurzelzuege, die nicht deutlich schlechter als der beste sind."""
    moves = list(board.legal_moves)
    if prev:
        moves.sort(key=lambda m: prev.get(m, -INF), reverse=True)
    else:
        moves = order_moves(board, moves, st, 0, None)

    results = {}
    best = -INF
    for m in moves:
        lower = best - ROOT_MARGIN if best > -INF // 2 else -INF
        board.push(m)
        sc = -negamax(board, depth - 1, -INF, -lower, 1, st)
        board.pop()
        if sc > lower:
            results[m] = sc
            if sc > best:
                best = sc
    return {m: s for m, s in results.items() if s >= best - ROOT_MARGIN}


# ══════════════════════════════════════════════════════════════════════════════
#  MENSCHLICHE ZUGAUSWAHL
# ══════════════════════════════════════════════════════════════════════════════
def human_pick(results, panic):
    best = max(results.values())
    if len(results) == 1 or best > MATE - 200:
        return max(results, key=results.get)

    # Gelegentlicher Aussetzer: ein "ganz ordentlich aussehender" Zug
    if rnd.random() < (BLUNDER_P_PANIC if panic else BLUNDER_P):
        others = [m for m in results if results[m] < best]
        if others:
            return rnd.choice(others)

    sigma = NOISE_SIGMA_PANIC if panic else NOISE_SIGMA
    noisy = {m: s + rnd.gauss(0, sigma) for m, s in results.items()}
    return max(noisy, key=noisy.get)


# ══════════════════════════════════════════════════════════════════════════════
#  ERÖFFNUNGSBUCH (kleine Auswahl, Schluessel = Zugfolge ab Startstellung)
# ══════════════════════════════════════════════════════════════════════════════
BOOK = {
    "": ["e2e4", "e2e4", "d2d4", "d2d4", "g1f3", "c2c4"],
    "e2e4": ["e7e5", "c7c5", "e7e6", "c7c6"],
    "e2e4 e7e5": ["g1f3", "g1f3", "f1c4", "b1c3"],
    "e2e4 e7e5 g1f3": ["b8c6", "b8c6", "g8f6"],
    "e2e4 e7e5 g1f3 b8c6": ["f1b5", "f1c4", "d2d4"],
    "e2e4 e7e5 g1f3 b8c6 f1b5": ["a7a6"],
    "e2e4 e7e5 g1f3 b8c6 f1c4": ["f8c5", "g8f6"],
    "e2e4 c7c5": ["g1f3", "b1c3"],
    "e2e4 c7c5 g1f3": ["d7d6", "b8c6", "e7e6"],
    "e2e4 e7e6": ["d2d4"],
    "e2e4 e7e6 d2d4": ["d7d5"],
    "e2e4 c7c6": ["d2d4"],
    "e2e4 c7c6 d2d4": ["d7d5"],
    "d2d4": ["d7d5", "g8f6"],
    "d2d4 d7d5": ["c2c4", "g1f3"],
    "d2d4 d7d5 c2c4": ["e7e6", "c7c6"],
    "d2d4 g8f6": ["c2c4", "g1f3"],
    "d2d4 g8f6 c2c4": ["e7e6", "g7g6"],
    "g1f3": ["d7d5", "g8f6"],
    "g1f3 d7d5": ["d2d4", "c2c4"],
    "g1f3 g8f6": ["c2c4", "g2g3"],
    "c2c4": ["e7e5", "g8f6", "c7c5"],
}


def book_move(board):
    if board.fullmove_number > 8 or board.chess960:
        return None
    key = " ".join(m.uci() for m in board.move_stack)
    # Buch nur gueltig, wenn die Partie wirklich ab Standardstellung begann
    if board.move_stack and board.root().fen() != chess.STARTING_FEN:
        return None
    opts = BOOK.get(key)
    if not opts:
        return None
    mv = chess.Move.from_uci(rnd.choice(opts))
    return mv if mv in board.legal_moves else None


# ══════════════════════════════════════════════════════════════════════════════
#  MENSCHLICHE BEDENKZEIT
# ══════════════════════════════════════════════════════════════════════════════
def human_think_time(board, remaining_ms, inc_ms, movestogo, n_legal, last_capture):
    """Liefert (Zielzeit, harte Obergrenze) in Sekunden."""
    t = remaining_ms / 1000.0
    inc = inc_ms / 1000.0
    fm = board.fullmove_number

    moves_left = movestogo or max(18, 45 - fm)
    base = t / moves_left + 0.75 * inc

    f = 1.0
    if fm <= 5:
        f *= 0.4            # Eroeffnung geht schnell
    elif fm <= 10:
        f *= 0.75
    if board.is_check():
        f *= 1.2
    if last_capture:
        f *= 0.7            # Rueckschlag meist offensichtlich
    f *= 0.7 + min(n_legal, 45) / 60.0     # mehr Zuege = mehr Nachdenken
    f *= math.exp(rnd.gauss(0, 0.45))      # natuerliche Streuung
    if rnd.random() < 0.07:
        f *= 2.2                           # kritischer Moment: tief nachdenken

    target = base * f
    target = min(target, t * 0.12 + inc * 0.8)
    if t < 10:
        target = min(target, 0.25 + inc * 0.5)
    target = max(target, 0.4 if t >= 20 else 0.1)
    target = min(target, max(0.05, t - 1.0))

    hard = min(max(target * 1.6, target + 0.3), max(0.1, t * 0.3))
    hard = max(hard, target * 0.5)
    return target, hard


# ══════════════════════════════════════════════════════════════════════════════
#  SUCHTHREAD
# ══════════════════════════════════════════════════════════════════════════════
class SearchThread(threading.Thread):
    def __init__(self, board, wtime=None, btime=None, winc=0, binc=0,
                 movetime=None, depth=None, movestogo=None, infinite=False,
                 last_capture=False, stop_event=None):
        super().__init__(daemon=True)
        self.root = board.copy()
        self.wtime, self.btime = wtime, btime
        self.winc, self.binc = winc or 0, binc or 0
        self.movetime, self.max_depth = movetime, depth
        self.movestogo = movestogo
        self.infinite = infinite
        self.last_capture = last_capture
        self.stop_event = stop_event or threading.Event()

    def sleep_until(self, t_start, target):
        """Wartet (unterbrechbar), bis `target` Sekunden vergangen sind."""
        rest = target - (time.time() - t_start)
        if rest > 0:
            self.stop_event.wait(timeout=rest)

    def finish(self, move):
        print(f"bestmove {move.uci() if move else '0000'}")
        sys.stdout.flush()

    def run(self):
        try:
            self._run()
        except Exception as e:
            print(f"search error: {e}", file=sys.stderr)
            sys.stderr.flush()
            moves = list(self.root.legal_moves)
            self.finish(moves[0] if moves else None)

    def _run(self):
        board = self.root
        t0 = time.time()
        legal = list(board.legal_moves)
        if not legal:
            self.finish(None)
            return

        us = board.turn
        remaining = self.wtime if us else self.btime
        inc = self.winc if us else self.binc
        fixed = self.movetime is not None or self.max_depth is not None or self.infinite
        simulate = not fixed and remaining is not None

        # Zeitplan bestimmen
        if self.movetime is not None:
            target = hard = self.movetime / 1000.0 * 0.95
        elif self.infinite:
            target, hard = 10 ** 6, 10 ** 6
        elif remaining is not None:
            target, hard = human_think_time(board, remaining, inc, self.movestogo,
                                            len(legal), self.last_capture)
        else:
            target = hard = rnd.uniform(0.5, 2.0)
        t_left = (remaining or 60_000) / 1000.0
        panic = t_left < 20

        # Eroeffnungsbuch
        if simulate:
            bm = book_move(board)
            if bm is not None:
                self.sleep_until(t0, min(target, rnd.uniform(0.3, 1.6)))
                self.finish(bm)
                return

        # Nur ein legaler Zug: fast sofort
        if len(legal) == 1 and not self.infinite:
            if simulate:
                self.sleep_until(t0, min(target, rnd.uniform(0.1, 0.5)))
            self.finish(legal[0])
            return

        # Tiefenlimit nach menschlichem Niveau
        npieces = len(board.piece_map())
        if self.max_depth:
            cap = self.max_depth
        else:
            cap = 3
            if npieces <= 12 and target >= 2.0:
                cap = 4
            if t_left < 8:
                cap = 2
            if self.infinite:
                cap = 6

        sboard = board.copy()
        st = SearchState(self.stop_event, t0 + hard)
        prev = None
        final = None
        depth = 1
        try:
            while depth <= cap:
                st.no_abort = (depth == 1)
                res = search_root(sboard, depth, st, prev)
                if not res:
                    break
                prev = res
                final = res
                el = time.time() - t0
                best_mv = max(res, key=res.get)
                print(f"info depth {depth} score cp {max(res.values())} "
                      f"time {int(el * 1000)} nodes {st.nodes} pv {best_mv.uci()}")
                sys.stdout.flush()
                if max(res.values()) > MATE - 200:
                    break
                if not self.infinite and el > target * 0.35:
                    break
                if self.stop_event.is_set():
                    break
                depth += 1
        except SearchAbort:
            pass

        if final:
            move = human_pick(final, panic)
        else:
            caps = [m for m in legal if board.is_capture(m)]
            move = rnd.choice(caps or legal)

        if self.infinite:
            self.stop_event.wait()
        elif simulate:
            self.sleep_until(t0, target)
        self.finish(move)


# ══════════════════════════════════════════════════════════════════════════════
#  UCI
# ══════════════════════════════════════════════════════════════════════════════
def uci_loop(thread_cls=None, extra_options=(), on_setoption=None, on_quit=None):
    thread_cls = thread_cls or SearchThread
    board = chess.Board()
    chess960 = False
    last_capture = False
    thread = None
    stop_event = threading.Event()

    def stop_search():
        nonlocal thread
        if thread and thread.is_alive():
            stop_event.set()
            thread.join(timeout=3.0)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            parts = line.split()
            if not parts:
                continue
            cmd = parts[0]

            if cmd == "uci":
                print("id name DarkOnEngine Human")
                print("id author Dark and Classic")
                print("option name UCI_Chess960 type check default false")
                for o in extra_options:
                    print(o)
                print("uciok")
            elif cmd == "isready":
                print("readyok")
            elif cmd == "setoption":
                low = line.lower()
                if "uci_chess960" in low:
                    chess960 = low.strip().endswith("true")
                elif on_setoption and "name" in parts:
                    ni = parts.index("name")
                    vi = parts.index("value") if "value" in parts else len(parts)
                    on_setoption(" ".join(parts[ni + 1:vi]),
                                 " ".join(parts[vi + 1:]), chess960)
            elif cmd == "ucinewgame":
                stop_search()
                board = chess.Board(chess960=chess960)
                last_capture = False
            elif cmd == "position":
                idx = 1
                last_capture = False
                if len(parts) > 1 and parts[1] == "startpos":
                    # FIX: Startpos ist immer die Standardstellung. (Das Original
                    # wuerfelte hier eine Chess960-Stellung aus, wodurch die
                    # folgenden Zuege ungueltig wurden.)
                    board = chess.Board(chess960=chess960)
                    idx = 2
                elif len(parts) > 1 and parts[1] == "fen":
                    j = parts.index("moves") if "moves" in parts else len(parts)
                    fen = " ".join(parts[2:j])
                    try:
                        board = chess.Board(fen, chess960=chess960)
                    except ValueError:
                        board = chess.Board(chess960=chess960)
                    idx = j
                if idx < len(parts) and parts[idx] == "moves":
                    for mv in parts[idx + 1:]:
                        try:
                            m = board.parse_uci(mv)
                        except ValueError:
                            break
                        last_capture = board.is_capture(m)
                        board.push(m)
            elif cmd == "go":
                kw = {}
                infinite = False
                i = 1
                while i < len(parts):
                    tok = parts[i]
                    if tok in ("wtime", "btime", "winc", "binc", "movetime",
                               "depth", "movestogo") and i + 1 < len(parts):
                        try:
                            kw[tok] = int(parts[i + 1])
                        except ValueError:
                            pass
                        i += 2
                    else:
                        if tok == "infinite":
                            infinite = True
                        i += 1
                stop_search()
                stop_event = threading.Event()
                thread = thread_cls(
                    board, wtime=kw.get("wtime"), btime=kw.get("btime"),
                    winc=kw.get("winc"), binc=kw.get("binc"),
                    movetime=kw.get("movetime"), depth=kw.get("depth"),
                    movestogo=kw.get("movestogo"), infinite=infinite,
                    last_capture=last_capture, stop_event=stop_event)
                thread.start()
            elif cmd == "stop":
                stop_search()
            elif cmd == "quit":
                stop_search()
                if on_quit:
                    on_quit()
                break
            sys.stdout.flush()
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.stderr.flush()
            continue


if __name__ == "__main__":
    uci_loop()
