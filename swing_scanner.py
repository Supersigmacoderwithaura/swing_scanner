#!/usr/bin/env python3
"""
Swing Scanner
-------------
Scans a list of stocks for swing-trade setups (pullbacks in uptrends and
volume breakouts), scores them, and prints a trade plan for each:
entry, stop, target, and position size.

Setup (one time, in Terminal):
    pip3 install yfinance pandas

Run:
    python3 swing_scanner.py
    python3 swing_scanner.py --watchlist mylist.txt --top 15
    python3 swing_scanner.py --account 25000 --risk 1

After each scan it writes swing_scan.html and opens it in your browser.

This is a research tool, not financial advice. It finds candidates;
you still need to read the chart and the news before trading.
"""

import argparse
import html
import math
import os
import sys
import webbrowser
from datetime import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing package. Run:  pip3 install yfinance pandas")
    sys.exit(1)

# ----------------------------- SETTINGS ------------------------------------
ACCOUNT_SIZE = 10_000      # your account size in dollars
RISK_PER_TRADE_PCT = 1.0   # percent of account risked per trade
MIN_PRICE = 10             # skip stocks cheaper than this
MIN_AVG_VOLUME = 500_000   # skip thin stocks (20-day average shares)
STOP_ATR_MULT = 1.5        # stop distance = this many ATRs below entry
TARGET_R = 2.0             # target = this many times the risk
MIN_SCORE = 50             # hide candidates scoring below this

DEFAULT_UNIVERSE = """
AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO AMD NFLX CRM ORCL ADBE INTC QCOM
MU AMAT LRCX PANW CRWD SNOW SHOP UBER ABNB PLTR COIN SQ PYPL
JPM BAC WFC GS MS C V MA AXP BLK SCHW
UNH JNJ LLY PFE MRK ABBV TMO ISRG VRTX REGN MRNA
XOM CVX COP SLB OXY EOG MPC
HD LOW WMT COST TGT NKE SBUX MCD CMG DIS
CAT DE BA GE HON LMT RTX UPS FDX
KO PEP PG MDLZ T VZ TMUS
FCX NEM LIN
""".split()
# ---------------------------------------------------------------------------


def load_universe(path):
    if not path:
        return DEFAULT_UNIVERSE
    with open(path) as f:
        text = f.read().replace(",", " ")
    tickers = [t.strip().upper() for t in text.split() if t.strip()]
    return sorted(set(tickers))


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def market_status(spy_df):
    """Is the broad market in an uptrend? Long setups work best when it is."""
    close = spy_df["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    if last > sma50 and sma50 > sma200:
        return "UPTREND", "Good conditions for long swing trades."
    if last > sma200:
        return "MIXED", "Be selective and size down."
    return "DOWNTREND", "Long setups fail more often here. Consider sitting out."


def analyze(ticker, df, spy_ret_63):
    """Return a dict describing the best setup for this ticker, or None."""
    df = df.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if len(df) < 210:
        return None

    close = df["Close"]
    last = close.iloc[-1]
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

    if last < MIN_PRICE or avg_vol < MIN_AVG_VOLUME:
        return None

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    rsi14 = rsi(close).iloc[-1]
    atr14 = atr(df).iloc[-1]
    rel_vol = df["Volume"].iloc[-1] / avg_vol
    ret_63 = last / close.iloc[-64] - 1
    rel_strength = ret_63 - spy_ret_63

    high20_prev = df["High"].iloc[-21:-1].max()   # prior 20-day high
    high20 = df["High"].iloc[-20:].max()
    high52 = df["High"].iloc[-252:].max()
    pullback_pct = (high20 - last) / high20 * 100
    extension_atr = (last - sma20) / atr14 if atr14 > 0 else 0

    uptrend = last > sma50 and sma50 > sma200

    # --- Setup detection ---
    setup = None
    if uptrend:
        if (
            3 <= pullback_pct <= 12
            and sma50 <= last <= sma20 + 0.5 * atr14
            and 35 <= rsi14 <= 55
        ):
            setup = "Pullback"
        elif last > high20_prev and rel_vol >= 1.3 and extension_atr < 3:
            setup = "Breakout"
    if setup is None:
        return None

    # --- Score (0-100) ---
    score = 0
    score += 15 if last > sma50 else 0
    score += 15 if sma50 > sma200 else 0
    score += max(0, min(20, rel_strength * 100))          # up to 20 for +20% vs SPY
    score += max(0, min(15, (rel_vol - 0.8) * 15))        # volume interest
    score += 25                                           # clean setup found
    near_high = last / high52
    score += max(0, min(10, (near_high - 0.85) / 0.15 * 10))  # near 52-week high
    score = round(min(score, 100))

    # --- Trade plan ---
    entry = last
    stop = entry - STOP_ATR_MULT * atr14
    risk_per_share = entry - stop
    target = entry + TARGET_R * risk_per_share
    risk_dollars = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    shares = math.floor(risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    # never put more than the whole account in one stock
    shares = min(shares, math.floor(ACCOUNT_SIZE / entry))

    return {
        "Ticker": ticker,
        "Setup": setup,
        "Score": score,
        "Price": round(entry, 2),
        "RSI": round(rsi14, 1),
        "RelVol": round(rel_vol, 2),
        "vsSPY_3m%": round(rel_strength * 100, 1),
        "Pullback%": round(pullback_pct, 1),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "Shares": shares,
        "Risk$": round(shares * risk_per_share, 2),
    }


def next_earnings(ticker):
    """Best-effort earnings date lookup. Returns a date string or ''."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            upcoming = [d for d in dates if d >= datetime.now().date()]
            if upcoming:
                return str(min(upcoming))
    except Exception:
        pass
    return ""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swing Scanner</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f5f5f4;--line:#ddd;--up:#0a7a3d;--mid:#b26b00;--down:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#9a9a9a;--card:#222;--line:#333;--up:#4cc38a;--mid:#e0a030;--down:#f2766e}}
body{font-family:-apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px;max-width:1200px;margin:auto}
h1{margin:0 0 4px;font-size:22px}.mut{color:var(--mut);font-size:13px}
.banner{background:var(--card);border-left:5px solid var(--c);padding:12px 16px;border-radius:6px;margin:16px 0}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{cursor:pointer;color:var(--mut);font-weight:600;user-select:none}th:first-child,td:first-child,td.l,th.l{text-align:left}
td a{color:inherit;font-weight:700}.s{display:inline-block;padding:2px 8px;border-radius:10px;background:var(--card);font-weight:700}
.hi{color:var(--up)}.md{color:var(--mid)}.warn{color:var(--down);font-weight:600}
</style></head><body>
<h1>Swing Scanner</h1>
<div class="mut">Scanned __WHEN__ &middot; last bar __BAR__ &middot; SPY 3-month __SPY__</div>
<div class="banner" style="--c:__COLOR__"><b>Market: __STATUS__.</b> __NOTE__</div>
__BODY__
<p class="mut">Click a column header to sort. Click a ticker to open its chart. This is a list of candidates, not advice:
check the chart and news first, and be careful holding through earnings. Shares and risk use the account size and risk % you set.</p>
<script>
document.querySelectorAll("th").forEach((th,i)=>th.onclick=()=>{
 const t=th.closest("table"),b=t.tBodies[0],d=th.dataset.d=th.dataset.d=="1"?"-1":"1";
 [...b.rows].sort((x,y)=>{let a=x.cells[i].dataset.v,c=y.cells[i].dataset.v,n=parseFloat(a),m=parseFloat(c);
 return (isNaN(n)||isNaN(m)?a.localeCompare(c):n-m)*d}).forEach(r=>b.appendChild(r))});
</script></body></html>"""


def write_html(out, status, note, bar_date, spy_ret, path="swing_scan.html"):
    color = {"UPTREND": "#0a7a3d", "MIXED": "#b26b00", "DOWNTREND": "#b3261e"}.get(status, "#888")
    if out is None or out.empty:
        body = "<p>No setups met your criteria. That is a normal outcome; don't force trades.</p>"
    else:
        head = "".join(
            f'<th class="{"l" if c in ("Ticker","Setup") else ""}">{html.escape(c)}</th>' for c in out.columns
        )
        rows = []
        for _, r in out.iterrows():
            cells = []
            for c in out.columns:
                v = str(r[c])
                if c == "Ticker":
                    t = html.escape(v)
                    cell = f'<a href="https://www.tradingview.com/symbols/{t}/" target="_blank" rel="noopener">{t}</a>'
                elif c == "Score":
                    cls = "hi" if r[c] >= 75 else "md" if r[c] >= 60 else ""
                    cell = f'<span class="s {cls}">{html.escape(v)}</span>'
                elif c == "Earnings" and v:
                    cell = f'<span class="warn">{html.escape(v)}</span>'
                else:
                    cell = html.escape(v)
                cls = ' class="l"' if c in ("Ticker", "Setup") else ""
                cells.append(f'<td{cls} data-v="{html.escape(v)}">{cell}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        body = f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    page = (
        PAGE.replace("__WHEN__", datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__BAR__", str(bar_date))
        .replace("__SPY__", f"{spy_ret * 100:.1f}%")
        .replace("__COLOR__", color)
        .replace("__STATUS__", html.escape(status))
        .replace("__NOTE__", html.escape(note))
        .replace("__BODY__", body)
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return os.path.abspath(path)


def main():
    global ACCOUNT_SIZE, RISK_PER_TRADE_PCT
    parser = argparse.ArgumentParser(description="Swing trade scanner")
    parser.add_argument("--watchlist", help="text file of tickers (space/comma/newline separated)")
    parser.add_argument("--top", type=int, default=15, help="how many results to show")
    parser.add_argument("--account", type=float, default=ACCOUNT_SIZE)
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE_PCT, help="percent risked per trade")
    parser.add_argument("--no-open", action="store_true", help="do not open the web page automatically")
    parser.add_argument("--min-score", type=int, default=MIN_SCORE)
    args = parser.parse_args()

    ACCOUNT_SIZE = args.account
    RISK_PER_TRADE_PCT = args.risk

    tickers = load_universe(args.watchlist)
    if "SPY" not in tickers:
        download_list = tickers + ["SPY"]
    else:
        download_list = tickers

    print(f"Downloading {len(download_list)} tickers...")
    data = yf.download(
        download_list,
        period="2y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data is None or data.empty:
        print("No data came back. Check your internet connection and try again.")
        sys.exit(1)

    # --- Market check ---
    spy = data["SPY"].dropna(subset=["Close"])
    status, note = market_status(spy)
    spy_ret_63 = spy["Close"].iloc[-1] / spy["Close"].iloc[-64] - 1
    print(f"\nMarket (SPY): {status}. {note}")
    print(f"Last bar: {spy.index[-1].date()}  |  SPY 3-month return: {spy_ret_63 * 100:.1f}%")
    print("(If you run this during market hours, the last bar is still forming.)\n")

    # --- Scan ---
    results = []
    for t in tickers:
        try:
            if t not in data.columns.get_level_values(0):
                continue
            row = analyze(t, data[t], spy_ret_63)
            if row and row["Score"] >= args.min_score:
                results.append(row)
        except Exception as e:
            print(f"  skipped {t}: {e}")

    bar_date = spy.index[-1].date()
    if not results:
        print("No setups met your criteria today. That is a normal outcome; don't force trades.")
        page_path = write_html(None, status, note, bar_date, spy_ret_63)
        if not args.no_open:
            webbrowser.open("file://" + page_path)
        return

    out = pd.DataFrame(results).sort_values("Score", ascending=False).head(args.top)

    # Earnings check only on the shortlist (slow, so limited)
    out["Earnings"] = [next_earnings(t) for t in out["Ticker"]]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))

    filename = f"swing_scan_{datetime.now():%Y-%m-%d}.csv"
    out.to_csv(filename, index=False)
    print(f"\nSaved to {filename}")
    page_path = write_html(out, status, note, bar_date, spy_ret_63)
    print(f"Web page: {page_path}")
    if not args.no_open:
        webbrowser.open("file://" + page_path)
    print(
        "\nBefore trading any of these: open the chart, read the latest news, "
        "and avoid holding through the Earnings date unless you mean to."
    )


if __name__ == "__main__":
    main()