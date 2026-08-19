#!/usr/bin/env python3
"""三配置 (A/B/C) DSpark 对比报告：HTML + PDF。

因子设计：
  A = mem 0.85 / chunk 16384
  B = mem 0.90 / chunk 32768
  C = mem 0.85 / chunk 32768
  C vs A -> chunked-prefill 纯效应 (mem 固定 0.85)
  C vs B -> mem-fraction 纯效应 (chunk 固定 32768)
"""
import json, statistics as st, sys
from pathlib import Path

FA, FB, FC = sys.argv[1], sys.argv[2], sys.argv[3]
HTML, PDF = Path(sys.argv[4]), Path(sys.argv[5])
FONT = '/root/work/models/dsv4-runtime/NotoSansCJKsc-VF.ttf'

DA, DB, DC = (json.loads(Path(f).read_text()) for f in (FA, FB, FC))
RA, RB, RC = DA['results'], DB['results'], DC['results']

NAMES = {'A': 'A：mem 0.85 / chunk 16K', 'B': 'B：mem 0.90 / chunk 32K', 'C': 'C：mem 0.85 / chunk 32K'}


def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def f(v, n=2):
    return 'N/A' if v is None else f'{v:.{n}f}'


def groups(rs):
    o = {}
    for ctx in sorted({r['context_length_target'] for r in rs}):
        for c in [1, 8, 16]:
            q = sorted([r for r in rs if r['context_length_target'] == ctx and r['concurrency'] == c],
                       key=lambda r: r['repeat'])
            v = [x['gen_throughput_tok_s'] for x in q]
            m = (abs(v[0]) + abs(v[1])) / 2 if len(v) == 2 else 0
            o[(ctx, c)] = {
                'thr': mean(v), 'ar': mean([x['accept_rate'] for x in q]),
                'al': mean([x['accept_len'] for x in q]), 'ttft': mean([x['ttft_mean_ms'] for x in q]),
                'tpot': mean([x['tpot_mean_ms'] for x in q]), 'itl': mean([x['itl_mean_ms'] for x in q]),
                'var': abs(v[0] - v[1]) / m * 100 if m else 0,
                'vram': max(x['peak_vram_mib'] for x in q),
                'power': mean([x['average_gpu_power_w'] for x in q]),
                'err': sum(x['error_count'] for x in q), 'to': sum(x['timeout_count'] for x in q),
            }
    return o


GA, GB, GC = groups(RA), groups(RB), groups(RC)
K = sorted(GA)


def best(k):
    return max(('A', GA[k]['thr']), ('B', GB[k]['thr']), ('C', GC[k]['thr']), key=lambda x: x[1])[0]


wins = {n: sum(1 for k in K if best(k) == n) for n in 'ABC'}


def gpu_summary(rs):
    out = []
    for i in range(8):
        q = [r['gpu_metrics'][str(i)] for r in rs]
        out.append((i, mean([x['avg_util_pct'] for x in q]), mean([x['avg_memory_mib'] for x in q]) / 1024,
                    max(x['peak_memory_mib'] for x in q) / 1024, mean([x['avg_power_w'] for x in q])))
    return out


def stats(rs):
    c1 = [r for r in rs if r['concurrency'] == 1]
    return {
        'avg': mean([r['gen_throughput_tok_s'] for r in rs]),
        'peak': max(r['gen_throughput_tok_s'] for r in rs),
        'ar': mean([r['accept_rate'] for r in rs]), 'al': mean([r['accept_len'] for r in rs]),
        'vram': max(r['peak_vram_mib'] for r in rs) / 1024,
        'c1': mean([r['gen_throughput_tok_s'] for r in c1]),
        'c1ar': mean([r['accept_rate'] for r in c1]), 'c1al': mean([r['accept_len'] for r in c1]),
        'ok': sum(r['success_count'] for r in rs), 'err': sum(r['error_count'] for r in rs),
        'to': sum(r['timeout_count'] for r in rs),
        'hivar': sum(1 for k in K if groups(rs)[k]['var'] > 5),
    }


SA, SB, SC = stats(RA), stats(RB), stats(RC)

# ---------------- HTML ----------------
css = '''*{box-sizing:border-box}body{font-family:"Noto Sans CJK SC","Noto Sans SC",Arial,sans-serif;margin:0;background:#fff;color:#222}
main{max-width:1280px;margin:auto;padding:34px}h1{font-size:30px;margin:0 0 8px;color:#1F4E78}
h2{font-size:19px;margin:30px 0 10px;border-bottom:2px solid #1F4E78;padding-bottom:7px;color:#1F4E78}
.sub{color:#666;line-height:1.7}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}
.card{border:1px solid #A6A6A6;padding:16px;background:#fff}.card b{display:block;font-size:24px;margin-top:6px;color:#1F4E78}
table{border-collapse:collapse;width:100%;font-size:12px;margin:10px 0 22px}
th,td{border:1px solid #BFBFBF;padding:7px;text-align:right}th{background:#1F4E78;color:#fff}
td:first-child,th:first-child{text-align:left}tr:nth-child(even) td{background:#F2F2F2}
.up{font-weight:700;color:#1F4E78}.bad{font-weight:700;color:#9C0006}.win{background:#DDEBF7!important;font-weight:700}
.note{border-left:4px solid #1F4E78;padding:10px 14px;background:#F2F2F2;line-height:1.7}
.small{font-size:11px;color:#555}@media print{main{max-width:none;padding:12mm}table{page-break-inside:auto}tr{page-break-inside:avoid}}'''

rows = ''
for ctx, c in K:
    a, b, cc = GA[(ctx, c)], GB[(ctx, c)], GC[(ctx, c)]
    w = best((ctx, c))
    ca, cb = (cc['thr'] / a['thr'] - 1) * 100, (cc['thr'] / b['thr'] - 1) * 100
    rows += (f"<tr><td>{ctx//1024}K / C{c}</td>"
             f"<td class='{'win' if w=='A' else ''}'>{a['thr']:.2f}</td>"
             f"<td class='{'win' if w=='B' else ''}'>{b['thr']:.2f}</td>"
             f"<td class='{'win' if w=='C' else ''}'>{cc['thr']:.2f}</td>"
             f"<td class='{'up' if ca>0 else 'bad'}'>{ca:+.2f}%</td>"
             f"<td class='{'up' if cb>0 else 'bad'}'>{cb:+.2f}%</td>"
             f"<td>{a['ar']:.3f}</td><td>{b['ar']:.3f}</td><td>{cc['ar']:.3f}</td>"
             f"<td>{a['al']:.3f}</td><td>{b['al']:.3f}</td><td>{cc['al']:.3f}</td></tr>")

lat = ''
for ctx, c in K:
    a, b, cc = GA[(ctx, c)], GB[(ctx, c)], GC[(ctx, c)]
    lat += (f"<tr><td>{ctx//1024}K / C{c}</td><td>{a['ttft']:.1f}</td><td>{b['ttft']:.1f}</td><td>{cc['ttft']:.1f}</td>"
            f"<td>{a['tpot']:.3f}</td><td>{b['tpot']:.3f}</td><td>{cc['tpot']:.3f}</td>"
            f"<td class='{'bad' if a['var']>5 else ''}'>{a['var']:.2f}%</td>"
            f"<td class='{'bad' if b['var']>5 else ''}'>{b['var']:.2f}%</td>"
            f"<td class='{'bad' if cc['var']>5 else ''}'>{cc['var']:.2f}%</td></tr>")

gpur = ''
for aa, bb, ccg in zip(gpu_summary(RA), gpu_summary(RB), gpu_summary(RC)):
    gpur += (f"<tr><td>GPU{aa[0]}</td><td>{aa[1]:.1f}%</td><td>{bb[1]:.1f}%</td><td>{ccg[1]:.1f}%</td>"
             f"<td>{aa[2]:.2f}</td><td>{bb[2]:.2f}</td><td>{ccg[2]:.2f}</td>"
             f"<td>{aa[3]:.2f}</td><td>{bb[3]:.2f}</td><td>{ccg[3]:.2f}</td>"
             f"<td>{aa[4]:.1f}</td><td>{bb[4]:.1f}</td><td>{ccg[4]:.1f}</td></tr>")


def details(rs, label):
    z = ''
    for r in rs:
        z += (f"<tr><td>{r['context_length_target']//1024}K</td><td>{r['concurrency']}</td><td>{r['repeat']}</td>"
              f"<td>{r['gen_throughput_tok_s']:.2f}</td><td>{r['accept_rate']:.3f}</td><td>{r['accept_len']:.3f}</td>"
              f"<td>{r['ttft_mean_ms']:.1f}</td><td>{r['ttft_p95_ms']:.1f}</td><td>{r['tpot_mean_ms']:.3f}</td>"
              f"<td>{r['itl_mean_ms']:.3f}</td><td>{r['peak_vram_mib']/1024:.2f}</td>"
              f"<td>{r['average_gpu_power_w']:.1f}</td><td>{r['error_count']}/{r['timeout_count']}</td></tr>")
    return (f'<h2>配置 {label}：30 组详细结果</h2><table><thead><tr><th>Context</th><th>C</th><th>R</th>'
            f'<th>Gen tok/s</th><th>AR</th><th>AL</th><th>TTFT ms</th><th>TTFT p95</th><th>TPOT ms</th>'
            f'<th>ITL ms</th><th>Peak VRAM GiB</th><th>Power W/GPU</th><th>Err/TO</th></tr></thead><tbody>{z}</tbody></table>')


k128 = [(131072, c) for c in (1, 8, 16)]
c128a = mean([(GC[k]['thr'] / GA[k]['thr'] - 1) * 100 for k in k128])
c128b = mean([(GC[k]['thr'] / GB[k]['thr'] - 1) * 100 for k in k128])

doc = f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>DSpark 三配置因子分解对比</title><style>{css}</style></head><body><main>
<h1>DeepSeek V4 Flash · DSpark 三配置因子分解</h1>
<div class="sub">A：mem 0.85 / chunk 16384　　B：mem 0.90 / chunk 32768　　C：mem 0.85 / chunk 32768<br>
<b>C vs A</b> 固定 mem=0.85，隔离 <code>chunked-prefill-size</code> 的纯效应；<b>C vs B</b> 固定 chunk=32768，隔离 <code>mem-fraction-static</code> 的纯效应。</div>
<div class="cards">
<div class="card">最优组合数 A / B / C<b>{wins['A']} / {wins['B']} / {wins['C']}</b><span class="small">共 15 组</span></div>
<div class="card">全局均值吞吐 C<b>{SC['avg']:.1f}</b><span class="small">tok/s（A {SA['avg']:.1f} · B {SB['avg']:.1f}）</span></div>
<div class="card">C 峰值显存 / GPU<b>{SC['vram']:.2f}</b><span class="small">GiB（A {SA['vram']:.2f} · B {SB['vram']:.2f}）</span></div>
<div class="card">高波动组数 A / B / C<b>{SA['hivar']} / {SB['hivar']} / {SC['hivar']}</b><span class="small">波动 &gt;5% 的组</span></div>
</div>
<div class="note"><b>核心结论：</b>三轮各 30 组、900 请求，Error / Timeout 全为 0 / 0。
<b>128K 长上下文的性能回退源自 <code>mem-fraction-static=0.90</code>，而非 32K chunk</b> —— 固定 chunk=32768 时，仅把 mem 从 0.90 降到 0.85，128K 三档并发平均回升 <b>{c128b:+.2f}%</b>；而固定 mem=0.85 时，把 chunk 从 16K 提到 32K，128K 档仅变化 <b>{c128a:+.2f}%</b>。
配置 C 同时取得最高全局均值吞吐、最少高波动组数，且显存占用比 B 低 {SB['vram']-SC['vram']:.2f} GiB。</div>
<h2>吞吐与投机解码：三配置对比（含因子分解）</h2>
<table><thead><tr><th>场景</th><th>A Gen</th><th>B Gen</th><th>C Gen</th><th>C vs A<br>chunk 效应</th><th>C vs B<br>mem 效应</th>
<th>A AR</th><th>B AR</th><th>C AR</th><th>A AL</th><th>B AL</th><th>C AL</th></tr></thead><tbody>{rows}</tbody></table>
<h2>延迟与重复稳定性</h2>
<table><thead><tr><th>场景</th><th>A TTFT</th><th>B TTFT</th><th>C TTFT</th><th>A TPOT</th><th>B TPOT</th><th>C TPOT</th>
<th>A 波动</th><th>B 波动</th><th>C 波动</th></tr></thead><tbody>{lat}</tbody></table>
<h2>GPU0–7 跨组资源均值</h2>
<table><thead><tr><th>GPU</th><th>A Util</th><th>B Util</th><th>C Util</th><th>A Mem</th><th>B Mem</th><th>C Mem</th>
<th>A Peak</th><th>B Peak</th><th>C Peak</th><th>A Power</th><th>B Power</th><th>C Power</th></tr></thead><tbody>{gpur}</tbody></table>
<h2>汇总统计</h2>
<table><thead><tr><th>指标</th><th>A</th><th>B</th><th>C</th></tr></thead><tbody>
<tr><td>全局均值吞吐 tok/s</td><td>{SA['avg']:.2f}</td><td>{SB['avg']:.2f}</td><td>{SC['avg']:.2f}</td></tr>
<tr><td>峰值吞吐 tok/s</td><td>{SA['peak']:.2f}</td><td>{SB['peak']:.2f}</td><td>{SC['peak']:.2f}</td></tr>
<tr><td>单流 C1 均值 tok/s</td><td>{SA['c1']:.2f}</td><td>{SB['c1']:.2f}</td><td>{SC['c1']:.2f}</td></tr>
<tr><td>Accept Rate 全局均值</td><td>{SA['ar']:.4f}</td><td>{SB['ar']:.4f}</td><td>{SC['ar']:.4f}</td></tr>
<tr><td>Accept Len 全局均值</td><td>{SA['al']:.4f}</td><td>{SB['al']:.4f}</td><td>{SC['al']:.4f}</td></tr>
<tr><td>单流 C1 Accept Rate</td><td>{SA['c1ar']:.4f}</td><td>{SB['c1ar']:.4f}</td><td>{SC['c1ar']:.4f}</td></tr>
<tr><td>单流 C1 Accept Len</td><td>{SA['c1al']:.4f}</td><td>{SB['c1al']:.4f}</td><td>{SC['c1al']:.4f}</td></tr>
<tr><td>峰值显存 / GPU (GiB)</td><td>{SA['vram']:.2f}</td><td>{SB['vram']:.2f}</td><td>{SC['vram']:.2f}</td></tr>
<tr><td>高波动组数 (&gt;5%)</td><td>{SA['hivar']}</td><td>{SB['hivar']}</td><td>{SC['hivar']}</td></tr>
<tr><td>成功 / 错误 / 超时</td><td>{SA['ok']} / {SA['err']} / {SA['to']}</td><td>{SB['ok']} / {SB['err']} / {SB['to']}</td><td>{SC['ok']} / {SC['err']} / {SC['to']}</td></tr>
</tbody></table>
{details(RA,'A')}{details(RB,'B')}{details(RC,'C')}
<h2>测量口径</h2>
<p class="small">Gen Throughput 为组级聚合吞吐（成功请求 completion tokens / 组墙钟时间）。AR / AL 只解析该组对应的 SGLang <code>Decode batch</code> 日志字节区间。
TTFT 为请求发出到首个非空流式内容事件；TPOT / ITL 在首 token 后按 completion token 数归一化。GPU 指标由 nvidia-smi 每 0.5 秒采样，GPU0–7 分别记录。
波动为同一 Context / Concurrency 下两次完整重复的 Gen Throughput 相对差异，超过 5% 的组标红，表示需要额外重复验证而非隐藏。
每组 30 请求 + 5 warmup，输出 1024 tokens，temperature=0 / top_p=1.0 / top_k=1 / seed=42 / ignore_eos。</p>
</main></body></html>'''
HTML.write_text(doc, encoding='utf-8')

# ---------------- PDF ----------------
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

pdfmetrics.registerFont(TTFont('Noto', FONT))
F = 'Noto'
BLUE = colors.HexColor('#1F4E78')
GRAY = colors.HexColor('#555555')
LINE = colors.HexColor('#BFBFBF')
ALT = colors.HexColor('#F2F2F2')
WIN = colors.HexColor('#DDEBF7')
RED = colors.HexColor('#9C0006')

S = {'title': ParagraphStyle('t', fontName=F, fontSize=21, leading=27, textColor=BLUE),
     'h': ParagraphStyle('h', fontName=F, fontSize=13, leading=18, textColor=BLUE, spaceBefore=6, spaceAfter=4),
     'b': ParagraphStyle('b', fontName=F, fontSize=7.6, leading=11.5, textColor=GRAY),
     's': ParagraphStyle('s', fontName=F, fontSize=5.7, leading=7.3, textColor=colors.HexColor('#222222')),
     'sr': ParagraphStyle('sr', fontName=F, fontSize=5.7, leading=7.3, textColor=RED),
     'w': ParagraphStyle('w', fontName=F, fontSize=5.7, leading=7.3, textColor=colors.white)}


def P(x, s='s'):
    return Paragraph(str(x), S[s])


def frame(c, d):
    c.saveState()
    c.setStrokeColor(BLUE)
    c.line(10 * mm, 12 * mm, 287 * mm, 12 * mm)
    c.setFont(F, 6.5)
    c.setFillColor(GRAY)
    c.drawString(10 * mm, 7 * mm, 'DeepSeek V4 Flash · DSpark 三配置因子分解对比')
    c.drawRightString(287 * mm, 7 * mm, f'第 {d.page} 页')
    c.restoreState()


def table(data, widths, font=5.7, extra=None):
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [('FONTNAME', (0, 0), (-1, -1), F), ('FONTSIZE', (0, 0), (-1, -1), font),
             ('BACKGROUND', (0, 0), (-1, 0), BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
             ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ALT]),
             ('GRID', (0, 0), (-1, -1), .3, LINE), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 2.5),
             ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]
    if extra:
        style += extra
    t.setStyle(TableStyle(style))
    return t


pdf = SimpleDocTemplate(str(PDF), pagesize=landscape(A4), leftMargin=9 * mm, rightMargin=9 * mm,
                        topMargin=9 * mm, bottomMargin=15 * mm,
                        title='DSpark 3-Way Factor Comparison', author='Hermes Agent')

st_ = [P('DeepSeek V4 Flash · DSpark 三配置因子分解', 'title'), Spacer(1, 2 * mm),
       P('A：mem 0.85 / chunk 16384　　B：mem 0.90 / chunk 32768　　C：mem 0.85 / chunk 32768', 'b'),
       P('C vs A 固定 mem=0.85，隔离 chunked-prefill-size 纯效应；C vs B 固定 chunk=32768，隔离 mem-fraction-static 纯效应。', 'b'),
       Spacer(1, 4 * mm)]

cards = Table([[P('最优组合 A/B/C', 'b'), P('全局均值吞吐 C', 'b'), P('C 峰值显存/GPU', 'b'), P('高波动组 A/B/C', 'b')],
               [P(f"{wins['A']} / {wins['B']} / {wins['C']}", 'h'), P(f"{SC['avg']:.1f} tok/s", 'h'),
                P(f"{SC['vram']:.2f} GiB", 'h'), P(f"{SA['hivar']} / {SB['hivar']} / {SC['hivar']}", 'h')]],
              colWidths=[68 * mm] * 4)
cards.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), .5, LINE), ('INNERGRID', (0, 0), (-1, -1), .3, LINE),
                           ('BACKGROUND', (0, 0), (-1, 0), ALT), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
st_ += [cards, Spacer(1, 4 * mm),
        P(f'核心结论：三轮各 30 组、900 请求，Error / Timeout 全为 0 / 0。128K 长上下文的性能回退源自 mem-fraction-static=0.90 而非 32K chunk：'
          f'固定 chunk=32768 时，仅把 mem 由 0.90 降到 0.85，128K 三档并发平均回升 {c128b:+.2f}%；'
          f'固定 mem=0.85 时，chunk 由 16K 提到 32K，128K 档仅变化 {c128a:+.2f}%。'
          f'配置 C 取得最高全局均值吞吐 {SC["avg"]:.1f} tok/s、最少高波动组（{SC["hivar"]} 组），显存比 B 低 {SB["vram"]-SC["vram"]:.2f} GiB。', 'b'),
        P('吞吐与投机解码对比（蓝底为该场景最优）', 'h')]

hdr = ['场景', 'A Gen', 'B Gen', 'C Gen', 'C vs A', 'C vs B', 'A AR', 'B AR', 'C AR', 'A AL', 'B AL', 'C AL']
data = [[P(x, 'w') for x in hdr]]
winbg = []
for i, (ctx, c) in enumerate(K, start=1):
    a, b, cc = GA[(ctx, c)], GB[(ctx, c)], GC[(ctx, c)]
    ca, cb = (cc['thr'] / a['thr'] - 1) * 100, (cc['thr'] / b['thr'] - 1) * 100
    col = {'A': 1, 'B': 2, 'C': 3}[best((ctx, c))]
    winbg.append(('BACKGROUND', (col, i), (col, i), WIN))
    data.append([P(f'{ctx//1024}K/C{c}'), P(f(a['thr'])), P(f(b['thr'])), P(f(cc['thr'])),
                 P(f'{ca:+.2f}%', 's' if ca >= 0 else 'sr'), P(f'{cb:+.2f}%', 's' if cb >= 0 else 'sr'),
                 P(f(a['ar'], 3)), P(f(b['ar'], 3)), P(f(cc['ar'], 3)),
                 P(f(a['al'], 3)), P(f(b['al'], 3)), P(f(cc['al'], 3))])
st_ += [table(data, [20 * mm] + [22 * mm] * 3 + [23 * mm] * 2 + [17 * mm] * 6, 5.9, winbg), PageBreak()]

st_ += [P('延迟与重复稳定性', 'h')]
hdr = ['场景', 'A TTFT', 'B TTFT', 'C TTFT', 'A TPOT', 'B TPOT', 'C TPOT', 'A 波动', 'B 波动', 'C 波动']
data = [[P(x, 'w') for x in hdr]]
for ctx, c in K:
    a, b, cc = GA[(ctx, c)], GB[(ctx, c)], GC[(ctx, c)]
    data.append([P(f'{ctx//1024}K/C{c}'), P(f(a['ttft'], 1)), P(f(b['ttft'], 1)), P(f(cc['ttft'], 1)),
                 P(f(a['tpot'], 3)), P(f(b['tpot'], 3)), P(f(cc['tpot'], 3)),
                 P(f(a['var'], 2) + '%', 'sr' if a['var'] > 5 else 's'),
                 P(f(b['var'], 2) + '%', 'sr' if b['var'] > 5 else 's'),
                 P(f(cc['var'], 2) + '%', 'sr' if cc['var'] > 5 else 's')])
st_ += [table(data, [24 * mm] + [28 * mm] * 9, 6.2), Spacer(1, 4 * mm), P('汇总统计', 'h')]

rowsx = [('全局均值吞吐 tok/s', f(SA['avg']), f(SB['avg']), f(SC['avg'])),
         ('峰值吞吐 tok/s', f(SA['peak']), f(SB['peak']), f(SC['peak'])),
         ('单流 C1 均值 tok/s', f(SA['c1']), f(SB['c1']), f(SC['c1'])),
         ('Accept Rate 全局均值', f(SA['ar'], 4), f(SB['ar'], 4), f(SC['ar'], 4)),
         ('Accept Len 全局均值', f(SA['al'], 4), f(SB['al'], 4), f(SC['al'], 4)),
         ('单流 C1 Accept Rate', f(SA['c1ar'], 4), f(SB['c1ar'], 4), f(SC['c1ar'], 4)),
         ('单流 C1 Accept Len', f(SA['c1al'], 4), f(SB['c1al'], 4), f(SC['c1al'], 4)),
         ('峰值显存 / GPU (GiB)', f(SA['vram']), f(SB['vram']), f(SC['vram'])),
         ('高波动组数 (>5%)', SA['hivar'], SB['hivar'], SC['hivar']),
         ('成功 / 错误 / 超时', f"{SA['ok']}/{SA['err']}/{SA['to']}", f"{SB['ok']}/{SB['err']}/{SB['to']}",
          f"{SC['ok']}/{SC['err']}/{SC['to']}")]
data = [[P(x, 'w') for x in ['指标', 'A', 'B', 'C']]] + [[P(r[0]), P(r[1]), P(r[2]), P(r[3])] for r in rowsx]
st_ += [table(data, [70 * mm, 60 * mm, 60 * mm, 60 * mm], 6.4), PageBreak()]

st_ += [P('GPU0–7 跨组资源均值', 'h')]
hdr = ['GPU', 'A Util', 'B Util', 'C Util', 'A Mem', 'B Mem', 'C Mem', 'A Peak', 'B Peak', 'C Peak', 'A W', 'B W', 'C W']
data = [[P(x, 'w') for x in hdr]]
for aa, bb, ccg in zip(gpu_summary(RA), gpu_summary(RB), gpu_summary(RC)):
    data.append([P(f'GPU{aa[0]}'), P(f(aa[1], 1) + '%'), P(f(bb[1], 1) + '%'), P(f(ccg[1], 1) + '%'),
                 P(f(aa[2])), P(f(bb[2])), P(f(ccg[2])), P(f(aa[3])), P(f(bb[3])), P(f(ccg[3])),
                 P(f(aa[4], 1)), P(f(bb[4], 1)), P(f(ccg[4], 1))])
st_ += [table(data, [18 * mm] + [21.5 * mm] * 12, 6.2), PageBreak()]

for label, rs in [('A：mem 0.85 / chunk 16384', RA), ('B：mem 0.90 / chunk 32768', RB), ('C：mem 0.85 / chunk 32768', RC)]:
    st_ += [P(f'配置 {label} · 30 组详细结果', 'h')]
    hdr = ['Ctx', 'C', 'R', 'Gen', 'AR', 'AL', 'TTFT mean/p50/p95', 'TPOT', 'ITL', 'Peak GiB', 'Power', 'OK', 'Err/TO']
    data = [[P(x, 'w') for x in hdr]]
    for r in rs:
        data.append([P(f"{r['context_length_target']//1024}K"), P(r['concurrency']), P(r['repeat']),
                     P(f(r['gen_throughput_tok_s'])), P(f(r['accept_rate'], 3)), P(f(r['accept_len'], 3)),
                     P(f"{f(r['ttft_mean_ms'],1)}/{f(r['ttft_p50_ms'],1)}/{f(r['ttft_p95_ms'],1)}"),
                     P(f(r['tpot_mean_ms'], 3)), P(f(r['itl_mean_ms'], 3)), P(f(r['peak_vram_mib'] / 1024)),
                     P(f(r['average_gpu_power_w'], 1)), P(r['success_count']),
                     P(f"{r['error_count']}/{r['timeout_count']}")])
    st_ += [table(data, [13 * mm, 9 * mm, 9 * mm, 20 * mm, 14 * mm, 14 * mm, 40 * mm, 18 * mm, 18 * mm,
                         22 * mm, 20 * mm, 14 * mm, 18 * mm], 5.5), PageBreak()]

st_ += [P('结论与测量口径', 'h'),
        P('因子分解表明：chunked-prefill-size 由 16384 提升到 32768 在中短上下文与高并发场景带来正向收益，且不是 128K 回退的原因；'
          'mem-fraction-static 由 0.85 提升到 0.90 会压缩可用 KV cache 余量，在 128K 长上下文下显著降低 Accept Rate 与吞吐。'
          '因此推荐配置 C（mem 0.85 + chunk 32768）：全场景均衡最佳、重复稳定性最好、显存占用低于 B。', 'b'),
        P('Gen 为组级聚合吞吐；AR/AL 来自对应组的服务日志字节区间；TTFT/TPOT/ITL 来自流式响应；'
          'GPU 指标每 0.5 秒采样；波动为两次完整重复的吞吐相对差异，>5% 标红提示需追加验证。', 'b')]

pdf.build(st_, onFirstPage=frame, onLaterPages=frame)
print(HTML, HTML.stat().st_size, PDF, PDF.stat().st_size)
