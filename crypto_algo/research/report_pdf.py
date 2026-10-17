"""Build a phone-friendly Hebrew PDF of the bot's P&L report from the CSVs written by research/report_bot.py.

Usage:
    python -m research.report_bot data/report_bot
    python research/report_pdf.py data/report_bot report.html
    chrome --headless --no-pdf-header-footer --print-to-pdf=report.pdf report.html
"""
import csv
import math
import sys
from datetime import date
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

rd = lambda name: list(csv.DictReader(open(SRC / name, encoding="utf-8")))
summary = {r["series"]: r for r in rd("summary.csv")}
equity = rd("equity.csv")
yearly = rd("yearly.csv")
coins = rd("coins.csv")
cy = rd("coin_yearly.csv")
niches = rd("niches.csv")
facts = {r["key"]: r["value"] for r in rd("facts.csv")}

C_BOT, C_HOLD, C_BTC = "#1d4ed8", "#9ca3af", "#d97706"
POS, NEG, MUTED, FG, LINE = "#15803d", "#b91c1c", "#6b7280", "#111827", "#e5e7eb"


def money(v, signed=False):
    v = float(v)
    s = ("+" if v > 0 else "−" if v < 0 else "") if signed else ("−" if v < 0 else "")
    return f'<span class="n">\u2066{s}${abs(v):,.0f}\u2069</span>'


def pct(v, signed=True, digits=1):
    v = float(v)
    s = ("+" if v > 0 else "−" if v < 0 else "") if signed else ("−" if v < 0 else "")
    return f'<span class="n">\u2066{s}{abs(v) * 100:,.{digits}f}%\u2069</span>'


def cls(v):
    v = float(v)
    return "pos" if v > 0 else "neg" if v < 0 else ""


def short_money(v):
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def line_chart(rows, series, log, width=760, height=300, yfmt=short_money, ticks=None):
    dates = [date.fromisoformat(r["date"]) for r in rows]
    d0, d1 = dates[0].toordinal(), dates[-1].toordinal()
    vals = [float(r[k]) for r in rows for k, _, _, _ in series]
    lo, hi = min(vals), max(vals)
    if log:
        lo, hi = math.log10(lo * 0.85), math.log10(hi * 1.15)
        ty = lambda v: math.log10(v)
    else:
        lo, hi = min(lo * 1.05, -0.05), 0.0
        ty = lambda v: v
    m = dict(l=58, r=150, t=12, b=26)
    X = lambda d: m["l"] + (d.toordinal() - d0) / (d1 - d0) * (width - m["l"] - m["r"])
    Y = lambda v: m["t"] + (hi - ty(v)) / (hi - lo) * (height - m["t"] - m["b"])
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Liberation Sans, DejaVu Sans, sans-serif" font-size="12" direction="ltr">']
    for t in ticks:
        if not (lo <= ty(t) <= hi):
            continue
        y = Y(t)
        out.append(f'<line x1="{m["l"]}" x2="{width - m["r"]}" y1="{y:.1f}" y2="{y:.1f}" stroke="{LINE}"/>')
        out.append(f'<text x="{m["l"] - 6}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTED}">{yfmt(t)}</text>')
    for yr in range(dates[0].year + 1, dates[-1].year + 1):
        x = X(date(yr, 1, 1))
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{height - m["b"]}" y2="{height - m["b"] + 4}" stroke="{MUTED}"/>')
        out.append(f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle" fill="{MUTED}">{yr}</text>')
    out.append(f'<line x1="{m["l"]}" x2="{width - m["r"]}" y1="{height - m["b"]}" y2="{height - m["b"]}" stroke="{MUTED}"/>')
    ends = []
    for key, label, color, w in reversed(series):
        pts = " ".join(f"{X(d):.1f},{Y(float(r[key])):.1f}" for d, r in zip(dates, rows))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>')
        ends.append([Y(float(rows[-1][key])), yfmt(float(rows[-1][key])), label, color])
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 15:
            ends[i][0] = ends[i - 1][0] + 15
    for y, value, label, color in ends:
        x0 = width - m["r"] + 8
        out.append(f'<text x="{x0}" y="{y + 4:.1f}" fill="{color}" font-weight="bold">{value}</text>')
        out.append(f'<text x="{x0 + 7.2 * len(value) + 6:.1f}" y="{y + 4:.1f}" fill="{color}">{label}</text>')
    out.append("</svg>")
    return "".join(out)


S_EQ = [("bot", "הבוט", C_BOT, 2.6), ("hold", "החזקה", C_HOLD, 1.8), ("btc", "ביטקוין", C_BTC, 1.6)]
eq_chart = line_chart(equity, S_EQ, True, ticks=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6])
S_DD = [("dd_bot", "הבוט", C_BOT, 2.2), ("dd_hold", "החזקה", C_HOLD, 1.6), ("dd_btc", "ביטקוין", C_BTC, 1.4)]
dd_chart = line_chart(equity, S_DD, False, height=230, yfmt=lambda v: ("−" if v < 0 else "") + f"{abs(v) * 100:.0f}%", ticks=[0, -0.2, -0.4, -0.6, -0.8])

b, h, t = summary["bot"], summary["hold"], summary["btc"]

year_rows = "".join(
    f'<tr><td class="b">{r["year"]}</td><td>{money(r["bot_start"])}</td><td class="b">{money(r["bot_end"])}</td>'
    f'<td class="b {cls(r["bot_change"])}">{pct(r["bot_change"])}</td><td class="{cls(r["bot_pnl"])}">{money(r["bot_pnl"], True)}</td>'
    f'<td>{pct(r["bot_dd"], False)}</td><td class="sep {cls(r["hold_change"])}">{pct(r["hold_change"])}</td><td>{pct(r["hold_dd"], False)}</td>'
    f'<td class="{cls(r["btc_change"])}">{pct(r["btc_change"])}</td></tr>'
    for r in yearly)

coins_sorted = sorted(coins, key=lambda r: -float(r["profit"]))
coin_rows = "".join(
    f'<tr><td class="b">{r["coin"]}</td><td>{r["niche"]}</td><td class="b {cls(r["profit"])}">{money(r["profit"], True)}</td>'
    f'<td class="{cls(r["profit_test"])}">{money(r["profit_test"], True)}</td><td>{pct(r["time_held"], False, 0)}</td>'
    f'<td>{r["trades"]}</td><td>{pct(r["win_rate"], False, 0)}</td><td class="pos">{pct(r["best_trade"], True, 0)}</td>'
    f'<td class="neg">{pct(r["worst_trade"], True, 0)}</td><td class="{cls(r["coin_change"])}">{pct(r["coin_change"], True, 0)}</td></tr>'
    for r in coins_sorted)

years = sorted({r["year"] for r in cy})
cyk = {(r["coin"], r["year"]): float(r["pp"]) for r in cy}


def heat(v):
    if v is None:
        return ""
    a = min(1.0, abs(v) / 0.5) * 0.55 + 0.08
    col = (21, 128, 61) if v > 0 else (185, 28, 28)
    txt = "0" if abs(v) < 0.005 else f'{"+" if v > 0 else "−"}{abs(v) * 100:.0f}%'
    return f'<td class="h" style="background: rgba({col[0]},{col[1]},{col[2]},{a:.2f})"><span class="n">{txt}</span></td>'


heat_rows = "".join(
    f'<tr><td class="b">{r["coin"]}</td>' + "".join(heat(cyk.get((r["coin"], y))) if (r["coin"], y) in cyk else "<td></td>" for y in years) + "</tr>"
    for r in coins_sorted)
niche_rows = "".join(
    f'<tr><td class="b">{r["niche"]}</td><td class="b {cls(r["profit"])}">{money(r["profit"], True)}</td><td class="{cls(r["share"])}">{pct(r["share"], True, 0)}</td>'
    f'<td class="{cls(r["profit_test"])}">{money(r["profit_test"], True)}</td><td class="small">{r["coins"]}</td></tr>'
    for r in sorted(niches, key=lambda r: -float(r["profit"])))

html = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><title>דוח רווח והפסד – בוט הקריפטו</title>
<style>
@page {{ size: A4; margin: 14mm 12mm; }}
body {{ font-family: "Liberation Sans", "DejaVu Sans", sans-serif; color: {FG}; font-size: 11.5pt; line-height: 1.45; margin: 0; }}
h1 {{ font-size: 22pt; margin: 0 0 2mm; }}
h2 {{ font-size: 14pt; margin: 6mm 0 2mm; padding-bottom: 1mm; border-bottom: 2px solid {LINE}; }}
.sub {{ color: {MUTED}; margin: 0 0 4mm; }}
.n {{ display: inline-block; direction: ltr; unicode-bidi: isolate; font-variant-numeric: tabular-nums; }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; margin: 3mm 0; }}
.kpi {{ border: 1px solid {LINE}; border-radius: 3mm; padding: 3mm 4mm; }}
.kpi .l {{ color: {MUTED}; font-size: 9.5pt; }}
.kpi .v {{ font-size: 19pt; font-weight: bold; margin: 1mm 0; }}
.kpi .c {{ color: {MUTED}; font-size: 9pt; }}
.legend {{ display: flex; gap: 5mm; font-size: 10pt; color: {MUTED}; margin: 1mm 0; }}
.legend i {{ display: inline-block; width: 6mm; height: 1.2mm; border-radius: 1mm; vertical-align: middle; margin-left: 1.5mm; }}
table {{ border-collapse: collapse; width: 100%; font-size: 10pt; }}
th {{ text-align: right; color: {MUTED}; font-weight: normal; font-size: 9pt; border-bottom: 1.5px solid {FG}; padding: 1.5mm 1.5mm; vertical-align: bottom; }}
td {{ text-align: right; padding: 1.6mm 1.5mm; border-bottom: 1px solid {LINE}; white-space: nowrap; }}
td.b {{ font-weight: bold; }} td.pos {{ color: {POS}; }} td.neg {{ color: {NEG}; }}
td.sep, th.sep {{ border-right: 2px solid {LINE}; }}
td.small {{ white-space: normal; font-size: 8.5pt; color: {MUTED}; }}
.heat td {{ text-align: center; padding: 1mm 0.8mm; font-size: 8.5pt; border-bottom: 1px solid #fff; }}
.heat td.b {{ text-align: right; }} .heat th {{ text-align: center; }}
.note {{ color: {MUTED}; font-size: 9pt; margin: 1.5mm 0; }}
ul.c {{ padding-right: 5mm; margin: 0; }} ul.c li {{ margin-bottom: 2.2mm; }}
.page {{ break-after: page; }}
.box {{ background: #f3f4f6; border-radius: 3mm; padding: 3mm 4mm; font-size: 10pt; }}
</style></head><body>

<section class="page">
<h1>דוח רווח והפסד – בוט הקריפטו</h1>
<p class="sub">הבוט בגרסת עד {float(facts["max_per_coin"]) * 100:.0f}% מהחשבון למטבע · {facts["coins_total"]} מטבעות · ספוט, קניות בלבד · אחרי עמלות · <span class="n">2018-01-01</span> עד <span class="n">{facts["data_end"]}</span></p>
<div class="kpis">
  <div class="kpi"><div class="l">התחלנו עם</div><div class="v">{money(b["start"])}</div><div class="c">1 בינואר 2018</div></div>
  <div class="kpi"><div class="l">סיימנו עם</div><div class="v" style="color:{POS}">{money(b["end"])}</div><div class="c">פי <span class="n">{float(b["multiple"]):.1f}</span> · רווח {money(float(b["end"]) - float(b["start"]), True)}</div></div>
  <div class="kpi"><div class="l">תשואה שנתית ממוצעת</div><div class="v">{pct(b["cagr"], True, 0)}</div><div class="c">החזקה: {pct(h["cagr"], True, 0)} · ביטקוין: {pct(t["cagr"], True, 0)}</div></div>
  <div class="kpi"><div class="l">ירידה מקסימלית מהשיא</div><div class="v" style="color:{NEG}">{pct(b["maxdd"], False)}</div><div class="c">מ-{money(b["dd_peak"])} (<span class="n">{b["dd_peak_date"]}</span>) ל-{money(b["dd_trough"])} (<span class="n">{b["dd_trough_date"]}</span>)</div></div>
  <div class="kpi"><div class="l">ירידה מקסימלית בהחזקה</div><div class="v" style="color:{NEG}">{pct(h["maxdd"], False)}</div><div class="c">מ-{money(h["dd_peak"])} ל-{money(h["dd_trough"])} ב-2018</div></div>
  <div class="kpi"><div class="l">מאז 2025 (שוק דובי)</div><div class="v">{pct(b["test_change"])}</div><div class="c">החזקה: {pct(h["test_change"])} · ביטקוין: {pct(t["test_change"])}</div></div>
</div>
<h2>צמיחת החשבון ($10,000 בהתחלה)</h2>
<div class="legend"><span><i style="background:{C_BOT}"></i>הבוט</span><span><i style="background:{C_HOLD}"></i>קנה והחזק – כל המטבעות</span><span><i style="background:{C_BTC}"></i>קנה והחזק – ביטקוין</span></div>
{eq_chart}
<p class="note">סולם לוגריתמי: כל קו אופקי הוא פי 3 או פי 10 בהון. "קנה והחזק" מחלק את הכסף שווה בין כל המטבעות שנסחרו באותו זמן.</p>
</section>

<section class="page">
<h2>שנה אחרי שנה</h2>
<table><thead><tr><th>שנה</th><th>התחלה</th><th>סוף</th><th>שינוי</th><th>רווח / הפסד</th><th>ירידה מקס׳ בשנה</th><th class="sep">קנה והחזק</th><th>ירידה מקס׳ (החזקה)</th><th>ביטקוין</th></tr></thead>
<tbody>{year_rows}</tbody></table>
<p class="note">2026 היא שנה חלקית (עד <span class="n">{facts["data_end"]}</span>). "ירידה מקס׳ בשנה" היא הנפילה הגדולה ביותר מהשיא בתוך אותה שנה.</p>
<h2>ירידה מהשיא לאורך הזמן</h2>
<div class="legend"><span><i style="background:{C_BOT}"></i>הבוט</span><span><i style="background:{C_HOLD}"></i>קנה והחזק</span><span><i style="background:{C_BTC}"></i>ביטקוין</span></div>
{dd_chart}
<p class="note">בכל רגע: כמה החשבון היה מתחת לשיא הקודם שלו.</p>
</section>

<section class="page">
<h2>רווח והפסד לפי מטבע</h2>
<table><thead><tr><th>מטבע</th><th>נישה</th><th>רווח / הפסד</th><th>מתוכו מאז 2025</th><th>זמן בפוזיציה</th><th>עסקאות</th><th>הצלחה</th><th>הטובה</th><th>הגרועה</th><th>שינוי המטבע</th></tr></thead>
<tbody>{coin_rows}</tbody></table>
<p class="note">רווח בדולרים מתוך החשבון שהתחיל ב-$10,000, כולל עמלות. מטבעות שנסחרו כשהחשבון כבר היה גדול (למשל PEPE מ-2023) נראים גדולים יותר בדולרים. "עסקה" היא תקופה רציפה שבה הבוט החזיק את המטבע. "שינוי המטבע" הוא כמה המחיר עצמו השתנה מאז שהתחיל להיסחר.</p>
</section>

<section class="page">
<h2>מי תרם ומתי</h2>
<p class="note">כמה אחוזים מהחשבון הוסיף (ירוק) או הוריד (אדום) כל מטבע בכל שנה. סכום כל עמודה הוא תשואת הבוט באותה שנה. תא ריק פירושו שהמטבע עוד לא נסחר.</p>
<table class="heat"><thead><tr><th>מטבע</th>{"".join(f"<th>{y}</th>" for y in years)}</tr></thead><tbody>{heat_rows}</tbody></table>
<h2>לפי נישה</h2>
<table><thead><tr><th>נישה</th><th>רווח / הפסד</th><th>חלק מהרווח</th><th>מתוכו מאז 2025</th><th>מטבעות</th></tr></thead><tbody>{niche_rows}</tbody></table>
</section>

<section>
<h2>מה מסיקים</h2>
<ul class="c">
<li><b>הבוט מגן בשנים רעות.</b> הוא ירד רק ב-{facts["years_down"]} מתוך {facts["years_total"]} שנים, ובשנה הגרועה ביותר ({facts["worst_year"]}) איבד {pct(facts["worst_year_change"])}. קנה והחזק ירד ב-{facts["hold_years_down"]} שנים, וב-2018 וב-2022 איבד יותר מ-70%.</li>
<li><b>רוב הרווח מגיע משנים חזקות מעטות.</b> השנה הטובה ביותר הייתה {facts["best_year"]} ({pct(facts["best_year_change"], True, 0)}). הבוט מחכה בצד, ותופס חזק כשיש שוק עולה. רק {pct(facts["months_up"], False, 0)} מהחודשים חיוביים, אבל החודשים הטובים גדולים מאוד.</li>
<li><b>צריך סבלנות.</b> הפרק הארוך ביותר שבו החשבון לא חזר לשיא שלו נמשך {facts["underwater_days"]} ימים, בערך שנתיים.</li>
<li><b>חלק מהמטבעות מרוויחים וחלק מפסידים, וזה חלק מהשיטה.</b> הבוט הרוויח ב-{facts["coins_profitable"]} מתוך {facts["coins_total"]} מטבעות. הכי הרבה הגיע מ-{facts["top_coin"]}. אי אפשר לדעת מראש מי ירוויח, ובמחקר, הוצאת מטבעות שהפסידו בשנה הקודמת הרעה את התוצאות.</li>
<li><b>רוב הזמן הכסף במזומן.</b> בממוצע רק {pct(facts["avg_invested"], False, 0)} מהחשבון במטבעות, ולכל היותר {pct(facts["max_invested"], False, 0)}. זה מה ששומר על הירידות קטנות.</li>
<li><b>בשוק הדובי האחרון הבוט שמר על הכסף, אבל לא הרוויח:</b> {pct(b["test_change"])} מאז 2025, לעומת {pct(h["test_change"])} בהחזקה.</li>
</ul>
<h2>איך הבוט עובד</h2>
<div class="box">
<p><b>כניסה:</b> 6 מערכות פריצה על נרות של 4 שעות. כל מערכת קונה כשהמחיר שובר את השיא של 50, 100 או 200 נרות, ומוכרת כשהוא שובר את השפל של 10 או 20 נרות. הן פועלות רק כשהסגירה היומית מעל EMA50.</p>
<p><b>בחירה:</b> מבין המטבעות שבמגמה, רק {facts["top_k"]} שעלו הכי הרבה ב-{facts["rank_days"]} הימים האחרונים.</p>
<p><b>גודל:</b> לכל היותר {pct(facts["max_per_coin"], False, 1)} מהחשבון במטבע אחד, ולכן לכל היותר {pct(facts["max_invested"], False, 0)} בשוק.</p>
<p><b>עלויות:</b> עמלה וסליפג' של {pct(facts["fee_per_side"], False, 2)} לכל צד. בלי מינוף ובלי שורטים.</p>
<p class="note">המספרים הם תוצאת backtest על נתוני Binance. המטבעות נבחרו היום, אחרי ששרדו, ולכן העבר מעט מנופח, גם לבוט וגם להחזקה. בסכומים גדולים, מסחר במטבעות קטנים היה מזיז את המחיר. תוצאות עבר אינן מבטיחות תוצאות עתידיות.</p>
</div>
</section>
</body></html>"""
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT, len(html))
