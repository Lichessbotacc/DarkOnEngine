#!/usr/bin/env python3
"""
Maia-Bot: nutzt die echten Maia-Netze (CSSLab/maia-chess), die auf Millionen
Lichess-Partien trainiert wurden, ueber lc0 mit `go nodes 1` (so wie bei den
Lichess-Bots maia1 / maia5 / maia9).

Die menschliche Bedenkzeit kommt aus darkon_human.py (liegt im selben Ordner).
Faellt lc0 aus oder fehlt, spielt automatisch die eingebaute Engine weiter.

Voraussetzungen:
  1. lc0 installiert (https://lczero.org), Pfad via Umgebungsvariable LC0_PATH
     oder UCI-Option Lc0Path. Ohne GPU: LC0_ARGS="--backend=eigen"
     (oder blas) setzen.
  2. pip install chess
  3. Maia-Gewichte: werden beim ersten Start automatisch nach ./maia_weights/
     geladen (maia-1100 ... maia-1900).
"""
import os
import sys
import time
import shlex
import shutil
import json
import zipfile
import platform
import threading
import urllib.request

import chess
import chess.engine

import darkon_human as dh

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {
    "lc0": os.environ.get("LC0_PATH", "lc0"),
    "level": int(os.environ.get("MAIA_LEVEL", "1700")),   # 1100 ... 1900
    "weights": "",
    "nodes": 1,
}
WEIGHT_URL = ("https://github.com/CSSLab/maia-chess/raw/master/"
              "maia_weights/maia-{lvl}.pb.gz")

_engine = None
_engine_failed = False
_lock = threading.Lock()


def log(msg):
    print(f"info string {msg}")
    sys.stdout.flush()


def weights_path():
    if CFG["weights"]:
        return CFG["weights"]
    lvl = min(1900, max(1100, round(CFG["level"] / 100) * 100))
    path = os.path.join(HERE, "maia_weights", f"maia-{lvl}.pb.gz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f"lade Maia-{lvl} Gewichte herunter ...")
        urllib.request.urlretrieve(WEIGHT_URL.format(lvl=lvl), path)
    return path


def _http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "maia-bot"})
    return urllib.request.urlopen(req, timeout=120)


def ensure_lc0():
    """Findet lc0 oder laedt es (nur Windows) automatisch herunter."""
    exe = "lc0.exe" if os.name == "nt" else "lc0"
    if os.path.isfile(CFG["lc0"]) or shutil.which(CFG["lc0"]):
        return CFG["lc0"]
    bindir = os.path.join(HERE, "lc0_bin")
    for root, _, files in os.walk(bindir):
        if exe in files:
            return os.path.join(root, exe)
    if os.name != "nt":
        raise RuntimeError("lc0 nicht gefunden (bitte lc0 installieren)")

    log("lade lc0 herunter (einmalig) ...")
    releases = json.load(_http("https://api.github.com/repos/LeelaChessZero/lc0/releases"))
    prefs = ("cpu-dnnl", "cpu-openblas", "cpu", "onnx")
    url = None
    for pref in prefs:
        for rel in releases:
            for a in rel.get("assets", []):
                n = a["name"].lower()
                if "windows" in n and n.endswith(".zip") and pref in n:
                    url = a["browser_download_url"]
                    break
            if url:
                break
        if url:
            break
    if not url:
        raise RuntimeError("kein passendes lc0-Windows-Paket gefunden")
    os.makedirs(bindir, exist_ok=True)
    zpath = os.path.join(bindir, "lc0.zip")
    with _http(url) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(bindir)
    os.remove(zpath)
    for root, _, files in os.walk(bindir):
        if exe in files:
            return os.path.join(root, exe)
    raise RuntimeError("lc0.exe nach dem Entpacken nicht gefunden")


def get_engine(chess960=False):
    """Startet lc0 mit Maia-Gewichten (einmalig). None, wenn nicht moeglich."""
    global _engine, _engine_failed
    with _lock:
        if _engine is not None:
            return _engine
        if _engine_failed:
            return None
        try:
            cmd = [ensure_lc0(), f"--weights={weights_path()}"]
            cmd += shlex.split(os.environ.get("LC0_ARGS", ""))
            _engine = chess.engine.SimpleEngine.popen_uci(cmd, timeout=60)
            if chess960:
                try:
                    _engine.configure({"UCI_Chess960": True})
                except Exception:
                    pass
            log(f"Maia-{CFG['level']} via lc0 geladen")
        except Exception as e:
            _engine_failed = True
            log(f"lc0/Maia nicht verfuegbar ({e}) - benutze eingebaute Engine")
            _engine = None
        return _engine


def close_engine():
    global _engine, _engine_failed
    with _lock:
        if _engine is not None:
            try:
                _engine.quit()
            except Exception:
                pass
        _engine = None
        _engine_failed = False


class MaiaThread(dh.SearchThread):
    def _run(self):
        eng = get_engine(self.root.chess960)
        if eng is None:
            return super()._run()

        board = self.root
        t0 = time.time()
        legal = list(board.legal_moves)
        if not legal:
            self.finish(None)
            return

        us = board.turn
        remaining = self.wtime if us else self.btime
        inc = self.winc if us else self.binc
        fixed = (self.movetime is not None or self.max_depth is not None
                 or self.infinite)
        simulate = not fixed and remaining is not None

        if remaining is not None and not fixed:
            target, _ = dh.human_think_time(board, remaining, inc, self.movestogo,
                                            len(legal), self.last_capture)
        else:
            target = 0.0
        if len(legal) == 1:
            target = min(target, dh.rnd.uniform(0.1, 0.5))

        try:
            res = eng.play(board.copy(), chess.engine.Limit(nodes=CFG["nodes"]))
            move = res.move
        except Exception as e:
            log(f"lc0-Fehler ({e}) - Fallback auf eingebaute Engine")
            close_engine()
            return super()._run()

        if move not in legal:
            move = legal[0]
        print(f"info depth 1 nodes {CFG['nodes']} pv {move.uci()}")
        sys.stdout.flush()

        if self.infinite:
            self.stop_event.wait()
        elif simulate:
            self.sleep_until(t0, target)
        self.finish(move)


def on_setoption(name, value, chess960):
    key = name.strip().lower()
    changed = True
    if key == "lc0path":
        CFG["lc0"] = value
    elif key == "maialevel":
        CFG["level"] = int(value)
        CFG["weights"] = ""
    elif key == "weightspath":
        CFG["weights"] = value
    elif key == "maianodes":
        CFG["nodes"] = max(1, int(value))
        changed = False
    else:
        changed = False
    if changed:
        close_engine()


EXTRA_OPTIONS = [
    "option name Lc0Path type string default lc0",
    "option name WeightsPath type string default ",
    "option name MaiaLevel type spin default 1700 min 1100 max 1900",
    "option name MaiaNodes type spin default 1 min 1 max 100",
]

if __name__ == "__main__":
    # Alles Noetige (lc0 + Gewichte) schon beim Start im Hintergrund vorbereiten
    threading.Thread(target=get_engine, daemon=True).start()
    dh.uci_loop(thread_cls=MaiaThread, extra_options=EXTRA_OPTIONS,
                on_setoption=on_setoption, on_quit=close_engine)
