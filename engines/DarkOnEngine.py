#!/usr/bin/env python3
"""
DarkOnEngine – UCI-kompatibler Schach-Motor
Menschlicher Spielstil: kohärente Pläne, kein sinnloses Hin-und-Her,
keine offensichtlichen Hänger, aber keine übermenschliche Stärke.
"""

import chess
import sys
import time
import threading
from collections import defaultdict, namedtuple
import random as rnd

INF         = 99_999_999
DRAW_PENALTY = 80_000   # Strafe für Dreifach-Wiederholung
WIN_THRESHOLD = 200     # Stellungsvorteil in cp

# ─── Figurenwerte ─────────────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20_000,
}

# ─── Positionstabellen (Weiß-Perspektive, a1=0) ───────────────────────────────
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

# König-PST: Mittelspiel (Sicherheit) und Endspiel (Aktivität)
KING_MIDDLE_PST = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

KING_END_PST = [
    -50,-30,-30,-30,-30,-30,-30,-50,
    -30,-10,  5,  5,  5,  5,-10,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,-10,  5,  5,  5,  5,-10,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

TTEntry = namedtuple("TTEntry", ["depth", "flag", "score", "best_move"])
# flag: 'EXACT' | 'LOWER' | 'UPPER'


# ═══════════════════════════════════════════════════════════════════════════════
#  PLAN-SYSTEM — Gibt der Engine eine kohärente strategische Ausrichtung
# ═══════════════════════════════════════════════════════════════════════════════
class Plan:
    DEVELOP    = "develop"     # Eröffnung: Figuren entwickeln
    ATTACK     = "attack"      # Königsangriff
    DEFEND     = "defend"      # Eigenen König schützen
    SIMPLIFY   = "simplify"    # Material-Vorteil durch Abtausch einwechseln
    COMPLICATE = "complicate"  # Komplikationen bei Material-Nachteil suchen
    IMPROVE    = "improve"     # Schlechteste Figur verbessern (Standardplan)


def assess_plan(board: chess.Board) -> str:
    """
    Analysiert die aktuelle Stellung und bestimmt den sinnvollsten Plan.
    Wird einmal pro 'go'-Befehl aufgerufen und bleibt für die gesamte
    Suche konstant — das macht das Spiel kohärenter.
    """
    us   = board.turn
    them = not us

    # ── Materialzählung ────────────────────────────────────────────────────
    mat_us   = sum(PIECE_VALUES.get(p.piece_type, 0)
                   for p in board.piece_map().values()
                   if p.color == us   and p.piece_type != chess.KING)
    mat_them = sum(PIECE_VALUES.get(p.piece_type, 0)
                   for p in board.piece_map().values()
                   if p.color == them and p.piece_type != chess.KING)
    diff = mat_us - mat_them

    # ── Eröffnungsphase: Entwicklung ──────────────────────────────────────
    if board.fullmove_number <= 12:
        home_rank = 0 if us == chess.WHITE else 7
        developed = sum(
            1 for pt in (chess.KNIGHT, chess.BISHOP)
            for sq in board.pieces(pt, us)
            if chess.square_rank(sq) != home_rank
        )
        if developed < 2:
            return Plan.DEVELOP

    # ── Material-basierte Pläne ────────────────────────────────────────────
    if diff >  300: return Plan.SIMPLIFY
    if diff < -300: return Plan.COMPLICATE

    # ── Königsangriff vorhanden? ───────────────────────────────────────────
    eking = board.king(them)
    if eking is not None:
        attacking = sum(
            1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[eking])
            if board.is_attacked_by(us, sq)
        )
        if attacking >= 3:
            return Plan.ATTACK

    # ── Eigener König in Gefahr? ───────────────────────────────────────────
    mking = board.king(us)
    if mking is not None:
        threatened = sum(
            1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[mking])
            if board.is_attacked_by(them, sq)
        )
        if threatened >= 3:
            return Plan.DEFEND

    return Plan.IMPROVE


def plan_bonus(board: chess.Board, plan: str) -> int:
    """
    Kleiner Bewertungsbonus (±5–25 cp) der den gewählten Plan unterstützt.
    Bewusst gering gehalten, damit taktische Überlegungen immer dominieren.
    """
    us   = board.turn
    them = not us
    score = 0

    if plan == Plan.ATTACK:
        eking = board.king(them)
        if eking is not None:
            for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                for sq in board.pieces(pt, us):
                    dist   = chess.square_distance(sq, eking)
                    score += max(0, (7 - dist)) * 2

    elif plan == Plan.DEFEND:
        mking = board.king(us)
        if mking is not None:
            for pt in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
                for sq in board.pieces(pt, us):
                    dist   = chess.square_distance(sq, mking)
                    score += max(0, (4 - dist)) * 3

    elif plan == Plan.SIMPLIFY:
        # Weniger Figuren auf dem Brett = besser für die führende Seite
        score += (32 - len(board.piece_map())) * 3

    elif plan == Plan.COMPLICATE:
        # Mehr Figuren = mehr Taktik = besser bei Material-Nachteil
        score -= (32 - len(board.piece_map())) * 2

    elif plan == Plan.DEVELOP:
        home_rank = 0 if us == chess.WHITE else 7
        for pt in (chess.KNIGHT, chess.BISHOP):
            for sq in board.pieces(pt, us):
                if chess.square_rank(sq) != home_rank:
                    score += 18
        # Frühzeitige Damenzüge bestrafen
        queen_home = chess.D1 if us == chess.WHITE else chess.D8
        if queen_home not in board.pieces(chess.QUEEN, us) and board.fullmove_number < 8:
            score -= 20

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  ZEITPLANUNG
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_think_time(remaining_ms: int, increment_ms: int = 0) -> float:
    """
    Berechnet eine menschlich wirkende Bedenkzeit mit Zufalls-Streuung.
    Berücksichtigt das Inkrement korrekt.
    """
    t   = remaining_ms  / 1000.0
    inc = increment_ms  / 1000.0

    # Mit Inkrement: pro Zug kommt `inc` Sekunden zurück
    if inc > 0:
        base  = t / 35 + inc * 0.75
        noise = rnd.uniform(-base * 0.3, base * 0.5)
        return max(0.05, base + noise)

    # Ohne Inkrement (stufenweise, mit Rauschen)
    if   t >= 3600: return rnd.uniform( 40, 200)
    elif t >= 1800: return rnd.uniform( 20, 110)
    elif t >= 1200: return rnd.uniform( 12,  55)
    elif t >=  600: return rnd.uniform(  5,  28)
    elif t >=  420: return rnd.uniform(  5,  20)
    elif t >=  300: return rnd.uniform(  5,  14)
    elif t >=  180: return rnd.uniform(  4,  10)
    elif t >=   60: return rnd.uniform(  3,   8)
    elif t >=   30: return rnd.uniform(  1,   2)
    elif t >=    5: return rnd.uniform(0.1, 1.5)
    else:           return 0.08


# ═══════════════════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════════════
def fast_board_key(board: chess.Board):
    """Kompakter Hash der aktuellen Stellung für die Transpositionstabelle."""
    return (board.board_fen(), board.turn, board.castling_xfen(), board.ep_square)


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """Most-Valuable-Victim / Least-Valuable-Attacker für die Zugsortierung."""
    score = 0
    if board.is_capture(move):
        if board.is_en_passant(move):
            victim_val = PIECE_VALUES[chess.PAWN]
        else:
            victim     = board.piece_at(move.to_square)
            victim_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker   = board.piece_at(move.from_square)
        att_val    = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        score     += victim_val * 10 - att_val
    if move.promotion:
        score += PIECE_VALUES[chess.QUEEN] // 2
    return score


def is_endgame(board: chess.Board) -> bool:
    """Einfache Endspiel-Erkennung anhand des Gesamtmaterials."""
    total = sum(
        PIECE_VALUES[p.piece_type]
        for p in board.piece_map().values()
        if p.piece_type != chess.KING
    )
    num_q = (len(board.pieces(chess.QUEEN, chess.WHITE))
             + len(board.pieces(chess.QUEEN, chess.BLACK)))
    num_r = (len(board.pieces(chess.ROOK,  chess.WHITE))
             + len(board.pieces(chess.ROOK,  chess.BLACK)))
    return total < 2600 or (num_q + num_r <= 2)


def king_pawn_shelter(board: chess.Board, color: chess.Color) -> int:
    """
    Berechnet den Bauernschutz vor dem König.
    Belohnt Bauern direkt vor dem König, bestraft offene Linien.
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    score     = 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    direction = 1 if color == chess.WHITE else -1

    for df in (-1, 0, 1):
        f = king_file + df
        if not (0 <= f <= 7):
            continue
        # Bauern auf der 1. und 2. Reihe vor dem König belohnen
        for dr in (1, 2):
            r = king_rank + direction * dr
            if not (0 <= r <= 7):
                continue
            sq    = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                score += 14 if dr == 1 else 6

        # Offene Linien vor dem König bestrafen
        has_own_pawn = any(
            chess.square_file(sq) == f
            for sq in board.pieces(chess.PAWN, color)
        )
        if not has_own_pawn:
            score -= 18

    return score


# ═══════════════════════════════════════════════════════════════════════════════
#  STELLUNGSBEWERTUNG
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(board: chess.Board, plan: str = Plan.IMPROVE) -> int:
    MATE_SCORE = 10_000_000

    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score    = 0
    material = 0
    endgame  = is_endgame(board)

    # ── Material + Positionstabellen ───────────────────────────────────────
    for pt, val in PIECE_VALUES.items():
        for sq in board.pieces(pt, chess.WHITE):
            material += val
            if pt in PST:
                score += PST[pt][sq]
            elif pt == chess.KING:
                score += (KING_END_PST if endgame else KING_MIDDLE_PST)[sq]

        for sq in board.pieces(pt, chess.BLACK):
            material -= val
            msq = chess.square_mirror(sq)
            if pt in PST:
                score -= PST[pt][msq]
            elif pt == chess.KING:
                score -= (KING_END_PST if endgame else KING_MIDDLE_PST)[msq]

    score += material

    # ── König-Sicherheit im Mittelspiel ───────────────────────────────────
    if not endgame:
        score += king_pawn_shelter(board, chess.WHITE)
        score -= king_pawn_shelter(board, chess.BLACK)

    # ── König-Aktivität im Endspiel ────────────────────────────────────────
    if endgame:
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        if wk is not None:
            score += KING_END_PST[wk]
            central = [chess.D4, chess.D5, chess.E4, chess.E5]
            if wk in central:
                score += 60
        if bk is not None:
            score -= KING_END_PST[chess.square_mirror(bk)]
            if bk in central:
                score -= 60

    # ── Rochade-Bonus ──────────────────────────────────────────────────────
    CASTLING_BONUS = 90
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk == chess.G1 and not board.has_kingside_castling_rights(chess.WHITE):
        score += CASTLING_BONUS
    elif wk == chess.C1 and not board.has_queenside_castling_rights(chess.WHITE):
        score += CASTLING_BONUS
    if bk == chess.G8 and not board.has_kingside_castling_rights(chess.BLACK):
        score -= CASTLING_BONUS
    elif bk == chess.C8 and not board.has_queenside_castling_rights(chess.BLACK):
        score -= CASTLING_BONUS

    # ── Läuferpaar ────────────────────────────────────────────────────────
    if len(board.pieces(chess.BISHOP, chess.WHITE)) == 2: score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) == 2: score -= 30

    # ── Freibauern ────────────────────────────────────────────────────────
    PASSED_BONUS = [0, 10, 20, 35, 60, 100, 140, 0]

    def is_passed(sq, color):
        file      = chess.square_file(sq)
        rank      = chess.square_rank(sq)
        direction = 1 if color == chess.WHITE else -1
        enemy     = not color
        r = rank + direction
        while 0 <= r <= 7:
            for df in (-1, 0, 1):
                f = file + df
                if 0 <= f <= 7:
                    p = board.piece_at(chess.square(f, r))
                    if p and p.piece_type == chess.PAWN and p.color == enemy:
                        return False
            r += direction
        return True

    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if is_passed(sq, chess.WHITE):
            score += PASSED_BONUS[chess.square_rank(sq)]
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        if is_passed(sq, chess.BLACK):
            score -= PASSED_BONUS[7 - chess.square_rank(sq)]

    # ── Türme ──────────────────────────────────────────────────────────────
    def has_pawn_on_file(file, color):
        return any(chess.square_file(s) == file for s in board.pieces(chess.PAWN, color))

    for sq in board.pieces(chess.ROOK, chess.WHITE):
        f = chess.square_file(sq)
        if not has_pawn_on_file(f, chess.WHITE):
            score += 25 if not has_pawn_on_file(f, chess.BLACK) else 15
        if chess.square_rank(sq) == 6:
            score += 28

    for sq in board.pieces(chess.ROOK, chess.BLACK):
        f = chess.square_file(sq)
        if not has_pawn_on_file(f, chess.BLACK):
            score -= 25 if not has_pawn_on_file(f, chess.WHITE) else 15
        if chess.square_rank(sq) == 1:
            score -= 28

    # ── Springer-Vorposten ─────────────────────────────────────────────────
    def is_outpost(sq, color):
        rank = chess.square_rank(sq)
        if color == chess.WHITE and rank < 3: return False
        if color == chess.BLACK and rank > 4: return False
        file  = chess.square_file(sq)
        enemy = not color
        dr    = 1 if color == chess.WHITE else -1
        r     = rank + dr
        while 0 <= r <= 7:
            for df in (-1, 1):
                f = file + df
                if 0 <= f <= 7:
                    p = board.piece_at(chess.square(f, r))
                    if p and p.piece_type == chess.PAWN and p.color == enemy:
                        return False
            r += dr
        return True

    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        if is_outpost(sq, chess.WHITE): score += 20
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        if is_outpost(sq, chess.BLACK): score -= 20

    # ── Vereinfachung bei Material-Vorteil ─────────────────────────────────
    if material >  200:
        score += 5 * (len(board.pieces(chess.QUEEN, chess.BLACK))
                      + len(board.pieces(chess.ROOK,  chess.BLACK)))
    if material < -200:
        score -= 5 * (len(board.pieces(chess.QUEEN, chess.WHITE))
                      + len(board.pieces(chess.ROOK,  chess.WHITE)))

    # ── Plan-Bonus ─────────────────────────────────────────────────────────
    score += plan_bonus(board, plan)

    return score if board.turn == chess.WHITE else -score


# ═══════════════════════════════════════════════════════════════════════════════
#  SUCHZUSTAND
# ═══════════════════════════════════════════════════════════════════════════════
class SearchState:
    def __init__(self, plan: str = Plan.IMPROVE):
        self.tt         = {}                  # Transpositionstabelle
        self.nodes      = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.history    = defaultdict(int)    # History-Heuristik
        self.plan       = plan
        # ── Anti-Shuffle: Zugpfad im aktuellen Suchast ───────────────────
        # Wird als Stack genutzt (append/pop bei board.push/pop)
        # path_moves[i] ist der Zug auf Tiefe i vom Wurzelknoten aus
        self.path_moves: list = []


class SearchAbort(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  QUIESCENCE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
def quiescence(board: chess.Board, alpha: int, beta: int,
               state: SearchState, stop_event: threading.Event) -> int:
    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1
    stand_pat = evaluate(board, state.plan)

    if stand_pat >= beta:  return beta
    if alpha < stand_pat:  alpha = stand_pat

    captures = [m for m in board.legal_moves
                if board.is_capture(m) or m.promotion]
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
        if score >= beta: return beta
        if score > alpha: alpha = score

    return alpha


# ═══════════════════════════════════════════════════════════════════════════════
#  NEGAMAX MIT ALPHA-BETA UND TRANSPOSITIONSTABELLE
# ═══════════════════════════════════════════════════════════════════════════════
def negamax(board: chess.Board, depth: int, alpha: int, beta: int,
            state: SearchState, stop_event: threading.Event) -> int:

    if stop_event.is_set():
        raise SearchAbort()
    if state.start_time and (time.time() - state.start_time) > state.time_limit:
        raise SearchAbort()

    state.nodes += 1

    if depth == 0:
        return quiescence(board, alpha, beta, state, stop_event)

    # ── Transpositionstabelle ──────────────────────────────────────────────
    key      = fast_board_key(board)
    tt_entry = state.tt.get(key)

    if tt_entry and tt_entry.depth >= depth:
        if   tt_entry.flag == 'EXACT': return tt_entry.score
        elif tt_entry.flag == 'LOWER': alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == 'UPPER': beta  = min(beta,  tt_entry.score)
        if alpha >= beta:
            return tt_entry.score

    alpha_orig = alpha
    beta_orig  = beta
    best_score = -INF
    best_move  = None
    moves      = list(board.legal_moves)

    # ── Zugsortierung ──────────────────────────────────────────────────────
    def move_priority(mv: chess.Move) -> int:
        # TT-Zug immer zuerst
        if tt_entry and tt_entry.best_move and mv == tt_entry.best_move:
            return -100_000
        p = 0
        if board.gives_check(mv):
            p -= 9_000
        if board.is_capture(mv):
            p -= mvv_lva_score(board, mv) * 5
        p -= state.history[(board.turn, mv.from_square, mv.to_square)]
        # Kleines Rauschen für menschliche Variation in gleichwertigen Zügen
        p += rnd.randint(0, 4)
        return p

    moves.sort(key=move_priority)

    for move in moves:
        if stop_event.is_set():
            raise SearchAbort()

        # ── Anti-Shuffle (erweiterter Check) ──────────────────────────────
        # path_moves[-2] ist der Zug der gleichen Seite zwei Halbzüge früher.
        # Wenn wir jetzt denselben Zug rückgängig machen wollen → überspringen.
        if (len(state.path_moves) >= 2
                and state.path_moves[-2].from_square == move.to_square
                and state.path_moves[-2].to_square   == move.from_square):
            continue

        mover = board.turn

        # ── Blunder-Check: Figur in angegriffenes Feld ziehen ─────────────
        # FIX: Originaler Code prüfte die falsche Seite (not board.turn vor push).
        # Korrekt: nach dem Push ist board.turn der Gegner, also prüfen wir
        # board.is_attacked_by(board.turn_after_push, to_square) = Gegner greift an.
        blunder_penalty = 0
        victim = board.piece_at(move.to_square)  # Figur die wir schlagen (falls vorhanden)
        mover_piece = board.piece_at(move.from_square)

        board.push(move)

        # ── Sofortige Terminal-Checks ──────────────────────────────────────
        if board.is_checkmate():
            board.pop()
            return 100_000 - depth   # Schnelleres Matt = höherer Score

        if board.is_stalemate():
            board.pop()
            continue   # Patt meiden wenn möglich

        # Dreifach-Wiederholung mit hoher Strafe, aber nicht sofort überspringen
        is_rep = board.can_claim_threefold_repetition()
        if is_rep:
            board.pop()
            continue   # Wiederholung konsequent vermeiden

        # ── Blunder-Penalty nach dem Push berechnen ────────────────────────
        # board.turn ist jetzt der Gegner (der unsere Figur schlagen kann)
        if mover_piece and mover_piece.piece_type != chess.PAWN:
            attackers_sq = list(board.attackers(board.turn, move.to_square))
            if attackers_sq:
                min_att_val = min(
                    PIECE_VALUES.get(board.piece_at(asq).piece_type, INF)
                    for asq in attackers_sq
                    if board.piece_at(asq)
                )
                our_piece_val = PIECE_VALUES.get(mover_piece.piece_type, 0)
                # Wenn unsere Figur deutlich mehr wert ist als der Angreifer
                # und wir keinen Ausgleich schlagen → Strafe
                if victim is None and our_piece_val > min_att_val + 80:
                    blunder_penalty = 350

        # ── Zug in Suchpfad eintragen, rekursiv suchen ────────────────────
        state.path_moves.append(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, state, stop_event)
        except SearchAbort:
            board.pop()
            state.path_moves.pop()
            raise
        finally:
            pass

        board.pop()
        state.path_moves.pop()

        # Strafe nach dem Pop anwenden
        score -= blunder_penalty

        if score > best_score:
            best_score = score
            best_move  = move

        if score > alpha:
            alpha = score
            if not board.is_capture(move):
                state.history[(mover, move.from_square, move.to_square)] += 2 ** depth

        if alpha >= beta:
            if not board.is_capture(move):
                state.history[(mover, move.from_square, move.to_square)] += 2 ** depth
            break

    # Wenn alle Züge übersprungen wurden (z.B. alle führen zu Wiederholungen)
    if best_score == -INF:
        # Gib einen neutralen Score zurück, statt -INF
        best_score = evaluate(board, state.plan)
        best_move  = None

    # ── TT-Speicherung ────────────────────────────────────────────────────
    if best_score >= beta_orig:   flag = 'LOWER'
    elif best_score <= alpha_orig: flag = 'UPPER'
    else:                          flag = 'EXACT'

    state.tt[key] = TTEntry(depth=depth, flag=flag,
                             score=best_score, best_move=best_move)
    return best_score


# ═══════════════════════════════════════════════════════════════════════════════
#  SUCHTHREAD MIT ITERATIVER VERTIEFUNG
# ═══════════════════════════════════════════════════════════════════════════════
class SearchThread(threading.Thread):
    def __init__(self, root_board: chess.Board,
                 wtime=None, btime=None, winc=0, binc=0,
                 movetime=None, max_depth=None, stop_event=None):
        super().__init__(daemon=True)
        self.root_board = root_board.copy()
        self.wtime      = wtime
        self.btime      = btime
        self.winc       = winc or 0
        self.binc       = binc or 0
        self.movetime   = movetime
        self.max_depth  = max_depth
        self.stop_event = stop_event or threading.Event()

        self.best_move     = None
        self.best_score    = None
        self.depth_reached = 0

        # ── Plan für diese Suche bestimmen ────────────────────────────────
        self.plan  = assess_plan(self.root_board)
        self.state = SearchState(plan=self.plan)

        # ── Anti-Mate-in-1: Sicherheitsnetz ──────────────────────────────
        # FIX: Ergebnis (safe_moves) wird korrekt gespeichert und genutzt.
        # FIX: opponent_mates_or_patt war früher undefiniert bei Patt.
        all_moves  = list(self.root_board.legal_moves)
        safe_moves = []

        for mv in all_moves:
            self.root_board.push(mv)
            opp_can_mate = False

            # Patt ist für den Ziehenden (falls führend) auch schlecht
            if not self.root_board.is_stalemate():
                for opp_mv in self.root_board.legal_moves:
                    self.root_board.push(opp_mv)
                    if self.root_board.is_checkmate():
                        opp_can_mate = True
                    self.root_board.pop()
                    if opp_can_mate:
                        break

            self.root_board.pop()

            if not opp_can_mate:
                safe_moves.append(mv)

        # FIX: Korrekte Zuweisung — wird in run() verwendet
        self.root_moves = safe_moves if safe_moves else all_moves

    def time_remaining_ms(self) -> int:
        if self.movetime:
            return self.movetime
        remaining = self.wtime if self.root_board.turn == chess.WHITE else self.btime
        if remaining is None:
            return 600
        inc = self.winc if self.root_board.turn == chess.WHITE else self.binc
        secs = calculate_think_time(remaining, inc)
        return int(max(50, secs * 1000))

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

                # ── Zugsortierung auf Wurzelebene ──────────────────────────
                root_key = fast_board_key(self.root_board)
                root_tt  = self.state.tt.get(root_key)

                def root_sort(mv: chess.Move) -> tuple:
                    if root_tt and root_tt.best_move and mv == root_tt.best_move:
                        return (-2, 0)
                    cap = 0 if self.root_board.is_capture(mv) else 1
                    return (cap, -mvv_lva_score(self.root_board, mv))

                # FIX: self.root_moves (aus __init__, gefiltert) verwenden
                ordered_moves = sorted(self.root_moves, key=root_sort)

                best_for_depth  = None
                best_score_for  = -INF
                candidates      = []  # (score, move) für Plan-Auswahl

                for mv in ordered_moves:
                    if self.stop_event.is_set():
                        break

                    self.root_board.push(mv)

                    # Dreifach-Wiederholung auf Wurzelebene ebenfalls meiden
                    if self.root_board.can_claim_threefold_repetition():
                        self.root_board.pop()
                        continue

                    try:
                        score = -negamax(
                            self.root_board, depth - 1,
                            -INF, INF,
                            self.state, self.stop_event,
                        )
                    except SearchAbort:
                        self.root_board.pop()
                        raise
                    finally:
                        pass

                    self.root_board.pop()

                    candidates.append((score, mv))

                    if score > best_score_for:
                        best_score_for = score
                        best_for_depth = mv

                    # Zeitcheck zwischen Wurzelzügen
                    if (time.time() - self.state.start_time) > self.state.time_limit:
                        break

                # ── Menschliche Zugauswahl ─────────────────────────────────
                # Unter Zügen innerhalb HUMAN_MARGIN cp des Bestens zufällig
                # (gewichtet) wählen — verhindert mechanisch-repetitives Spiel.
                if best_for_depth is not None and depth >= 3 and len(candidates) > 1:
                    HUMAN_MARGIN = 15  # cp — bewusst konservativ
                    near_best = [
                        (s, m) for s, m in candidates
                        if s >= best_score_for - HUMAN_MARGIN
                    ]
                    if len(near_best) > 1:
                        # Gewichtete Zufallsauswahl (bessere Züge wahrscheinlicher)
                        weights = [s - best_score_for + HUMAN_MARGIN + 1
                                   for s, _ in near_best]
                        total   = sum(weights)
                        if total > 0:
                            r, cum = rnd.uniform(0, total), 0
                            for (s, m), w in zip(near_best, weights):
                                cum += w
                                if r <= cum:
                                    best_for_depth = m
                                    best_score_for = s
                                    break

                if best_for_depth is not None:
                    self.best_move  = best_for_depth
                    self.best_score = best_score_for
                    elapsed = time.time() - self.state.start_time
                    nps     = int(self.state.nodes / elapsed) if elapsed > 0 else 0
                    print(
                        f"info depth {depth} "
                        f"score cp {best_score_for} "
                        f"time {int(elapsed * 1000)} "
                        f"nodes {self.state.nodes} "
                        f"nps {nps} "
                        f"pv {best_for_depth.uci()}"
                    )
                    sys.stdout.flush()

                if (time.time() - self.state.start_time) > self.state.time_limit:
                    break

                depth += 2  # Iterative Vertiefung in 2er-Schritten

        except SearchAbort:
            pass
        except Exception as e:
            print(f"Search error: {e}", file=sys.stderr)
            sys.stderr.flush()

        # ── Besten Zug ausgeben ───────────────────────────────────────────
        move = self.best_move
        if move is None:
            # Fallback: bester verfügbarer Zug ohne Suche
            moves = list(self.root_board.legal_moves)
            caps  = [m for m in moves if self.root_board.is_capture(m)]
            move  = (max(caps, key=lambda m: mvv_lva_score(self.root_board, m))
                     if caps else (moves[0] if moves else None))

        if move:
            print(f"bestmove {move.uci()}")
        else:
            print("bestmove 0000")
        sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  UCI-HAUPTSCHLEIFE
# ═══════════════════════════════════════════════════════════════════════════════
def uci_loop():
    board         = chess.Board()
    chess960_mode = False
    search_thread: SearchThread | None = None
    stop_event    = threading.Event()

    print("id name DarkOnEngine")
    print("id author Dark and Classic")
    print("option name UCI_Chess960 type check default false")
    print("uciok")
    sys.stdout.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line  = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd   = parts[0]

            if cmd == "uci":
                print("id name DarkOnEngine")
                print("id author Dark and Classic")
                print("option name UCI_Chess960 type check default false")
                print("uciok")
                sys.stdout.flush()

            elif cmd == "isready":
                print("readyok")
                sys.stdout.flush()

            elif cmd == "setoption":
                if "UCI_Chess960" in line:
                    chess960_mode = "true" in line.lower()

            elif cmd == "ucinewgame":
                if chess960_mode:
                    board = chess.Board.from_chess960_pos(rnd.randint(0, 959))
                else:
                    board = chess.Board()

            elif cmd == "position":
                idx = 1
                if len(parts) >= 2 and parts[1] == "startpos":
                    board = (chess.Board.from_chess960_pos(rnd.randint(0, 959))
                             if chess960_mode else chess.Board())
                    idx = 2
                elif len(parts) >= 2 and parts[1] == "fen":
                    if len(parts) >= 8:
                        fen = " ".join(parts[2:8])
                        try:
                            board = chess.Board(fen, chess960=chess960_mode)
                        except Exception:
                            board = chess.Board()
                        idx = 8
                if idx < len(parts) and parts[idx] == "moves":
                    for mv in parts[idx + 1:]:
                        try:
                            board.push_uci(mv)
                        except Exception:
                            pass

            elif cmd == "go":
                wtime = btime = winc = binc = movetime = depth_limit = None
                i = 1
                while i < len(parts):
                    tok = parts[i]
                    if tok in ("wtime", "btime", "winc", "binc",
                               "movetime", "depth") and i + 1 < len(parts):
                        val = int(parts[i + 1])
                        if   tok == "wtime":    wtime       = val
                        elif tok == "btime":    btime       = val
                        elif tok == "winc":     winc        = val
                        elif tok == "binc":     binc        = val
                        elif tok == "movetime": movetime    = val
                        elif tok == "depth":    depth_limit = val
                        i += 2
                    else:
                        i += 1

                # Laufende Suche stoppen
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=1.0)
                    stop_event.clear()

                # ── Erster Zug: aus sinnvollen Eröffnungszügen wählen ─────
                # Zentrums- und Entwicklungszüge bevorzugen, kein Zufall über
                # alle legalen Züge → menschlichere Eröffnung
                if board.fullmove_number == 1 and board.turn == chess.WHITE:
                    center_moves = [
                        m for m in board.legal_moves
                        if chess.square_file(m.to_square) in (2, 3, 4, 5)
                        and chess.square_rank(m.to_square) in (3, 4)
                    ]
                    mv = rnd.choice(center_moves if center_moves
                                    else list(board.legal_moves))
                    print(f"bestmove {mv.uci()}")
                    sys.stdout.flush()
                    continue

                # Zweiter Zug für Schwarz: Spiegelzug oder gute Eröffnung
                if board.fullmove_number == 1 and board.turn == chess.BLACK:
                    dev_moves = [
                        m for m in board.legal_moves
                        if chess.square_file(m.to_square) in (2, 3, 4, 5)
                        and chess.square_rank(m.to_square) in (3, 4)
                    ]
                    mv = rnd.choice(dev_moves if dev_moves
                                    else list(board.legal_moves))
                    print(f"bestmove {mv.uci()}")
                    sys.stdout.flush()
                    continue

                stop_event    = threading.Event()
                search_thread = SearchThread(
                    board,
                    wtime=wtime, btime=btime,
                    winc=winc or 0, binc=binc or 0,
                    movetime=movetime,
                    max_depth=depth_limit,
                    stop_event=stop_event,
                )
                search_thread.start()

            elif cmd == "stop":
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=2.0)

            elif cmd == "quit":
                if search_thread and search_thread.is_alive():
                    stop_event.set()
                    search_thread.join(timeout=2.0)
                break

        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.stderr.flush()
            break


if __name__ == "__main__":
    uci_loop()
