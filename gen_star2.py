#!/usr/bin/env python3
"""Generate the Revenue star-schema diagram (Fact_Revenue) as SVG in drawio table style."""

ROW_H = 40
HDR_H = 40
COL_TYPE_W = 60
FONT = "Trebuchet MS, Verdana, Arial, sans-serif"
CANVAS_W, CANVAS_H = 1260, 1020


def table_svg(x, y, name, rows):
    """rows: list of (type, colname, key) where key is 'PK', 'FK' or ''."""
    key_w = 45
    name_col_w = max([len(r[1]) for r in rows] + [len(name)]) * 8 + 40
    w = COL_TYPE_W + name_col_w + key_w
    h = HDR_H + ROW_H * len(rows)
    out = ["<g>"]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{HDR_H}" class="hdr"/>')
    out.append(f'<text x="{x+w/2}" y="{y+HDR_H/2+6}" text-anchor="middle" class="hdrtxt">{name}</text>')
    for i, (typ, col, key) in enumerate(rows):
        ry = y + HDR_H + i * ROW_H
        cls = "rowA" if i % 2 == 0 else "rowB"
        out.append(f'<rect x="{x}" y="{ry}" width="{w}" height="{ROW_H}" class="{cls}"/>')
        out.append(f'<line x1="{x+COL_TYPE_W}" y1="{ry}" x2="{x+COL_TYPE_W}" y2="{ry+ROW_H}" stroke="#9370DB"/>')
        out.append(f'<line x1="{x+COL_TYPE_W+name_col_w}" y1="{ry}" x2="{x+COL_TYPE_W+name_col_w}" y2="{ry+ROW_H}" stroke="#9370DB"/>')
        out.append(f'<text x="{x+8}" y="{ry+ROW_H/2+5}" class="txt">{typ}</text>')
        out.append(f'<text x="{x+COL_TYPE_W+8}" y="{ry+ROW_H/2+5}" class="txt">{col}</text>')
        out.append(f'<text x="{x+COL_TYPE_W+name_col_w+8}" y="{ry+ROW_H/2+5}" class="txt">{key}</text>')
    out.append("</g>")
    return "\n".join(out), w, h


def edge(x1, y1, x2, y2, label, mx=None, my=None):
    path = f"M {x1} {y1}"
    if mx is not None:
        path += f" L {mx} {my}"
        lxm, lym = (x1 + mx) / 2, (y1 + my) / 2
    else:
        lxm, lym = (x1 + x2) / 2, (y1 + y2) / 2
    path += f" L {x2} {y2}"
    return (f'<path d="{path}" class="edge"/>\n'
            f'<text x="{lxm}" y="{lym-6}" text-anchor="middle" class="lbl">"{label}"</text>')


fact_rows = [
    ("int", "revenue_id", "PK"), ("int", "date_id", "FK"), ("int", "content_id", "FK"),
    ("int", "user_id", "FK"), ("int", "payment_id", "FK"), ("int", "plan_id", "FK"),
    ("int", "device_id", "FK"), ("float", "amount", ""), ("float", "discount_amount", ""),
]
fact_svg, fw, fh = table_svg(480, 430, "Fact_Revenue", fact_rows)
parts = [fact_svg]
fx, fy, fcx = 480, 430, 480 + fw / 2

d_date, w, h = table_svg(510, 30, "Dim_Date", [("int", "date_id", "PK"), ("date", "full_date", ""), ("int", "month", ""), ("int", "quarter", ""), ("int", "year", "")])
d_content, w2, h2 = table_svg(60, 60, "Dim_Content", [("int", "content_id", "PK"), ("string", "title", ""), ("string", "genre", ""), ("string", "content_type", "")])
d_user, w3, h3 = table_svg(20, 480, "Dim_User", [("int", "user_id", "PK"), ("string", "username", ""), ("string", "subscription_tier", "")])
d_pay, w4, h4 = table_svg(80, 760, "Dim_Payment", [("int", "payment_id", "PK"), ("string", "payment_mode", ""), ("string", "payment_gateway", ""), ("string", "status", "")])
d_plan, w5, h5 = table_svg(830, 760, "Dim_Plan", [("int", "plan_id", "PK"), ("string", "plan_name", ""), ("float", "monthly_price", ""), ("string", "billing_cycle", "")])
d_device, w6, h6 = table_svg(950, 80, "Dim_Device", [("int", "device_id", "PK"), ("string", "device_type", ""), ("string", "os_version", "")])
parts += [d_date, d_content, d_user, d_pay, d_plan, d_device]

edges = [
    edge(fcx, 30 + h, fcx, fy, "occurs on"),
    edge(60 + w2, 60 + h2, fx + 40, fy, "generates", mx=60 + w2, my=fy - 60),
    edge(20 + w3, 480 + h3 / 2, fx, fy + 90, "subscribes"),
    edge(80 + w4 / 2, 760, fx + 40, fy + fh, "paid via", mx=80 + w4 / 2, my=fy + fh + 40),
    edge(830 + w5 / 2, 760, fx + fw - 40, fy + fh, "under plan", mx=830 + w5 / 2, my=fy + fh + 40),
    edge(960, 80 + h6, fx + fw, fy + 120, "watched via", mx=960, my=fy + 120),
]

svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}" font-family="{FONT}">
<style>
  .hdr{{fill:#ECECFF;stroke:#9370DB;stroke-width:1;}}
  .rowA{{fill:#ffffff;stroke:#9370DB;stroke-width:1;}}
  .rowB{{fill:#F1F1FF;stroke:#9370DB;stroke-width:1;}}
  .txt{{fill:#333333;font-size:16px;}}
  .hdrtxt{{fill:#333333;font-size:17px;font-weight:bold;}}
  .lbl{{fill:#333333;font-size:14px;}}
  .edge{{fill:none;stroke:#333333;stroke-width:1;}}
</style>
<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="white"/>
{"".join(edges)}
{"".join(parts)}
</svg>'''

with open("second_star_full.svg", "w") as f:
    f.write(svg)
print("wrote second_star_full.svg")
