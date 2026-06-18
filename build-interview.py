#!/usr/bin/env python3
"""One-off: convert the 4 hand-authored interview HTML pages into
data-driven shells (inline DATA + assets/interview.js + assets/common.css).

Body HTML is sliced verbatim from the original source (depth-counted div
matching), so rendered output is identical. Run from repo root:
    python3 assets/_build.py
"""
import re, json, os

ROOT = os.path.dirname(os.path.abspath(__file__))

PAGES = [
    ("transformer-interview.html", "#7c3aed", "rgba(124,58,237,0.06)", "rgba(124,58,237,0.1)"),
    ("rl-interview.html",          "#30a14e", "rgba(48,161,78,0.06)",  "rgba(48,161,78,0.1)"),
    ("sft-interview.html",         "#0071e3", "rgba(0,113,227,0.06)",  "rgba(0,113,227,0.1)"),
    ("agent-interview.html",       "#7c3aed", "rgba(124,58,237,0.06)", "rgba(124,58,237,0.1)"),
]

DIV_TOKEN = re.compile(r'<(/?)div\b[^>]*>')

def slice_body(s, start):
    """Return verbatim inner HTML of the .q-body whose opening tag ends at `start`."""
    depth = 1
    i = start
    while depth > 0:
        m = DIV_TOKEN.search(s, i)
        if not m:
            raise RuntimeError("unbalanced div in q-body")
        if m.group(1) == '/':
            depth -= 1
            if depth == 0:
                return s[start:m.start()]
        else:
            depth += 1
        i = m.end()
    return s[start:i]

def first(pattern, s, flags=re.S, group=1):
    m = re.search(pattern, s, flags)
    if not m:
        raise RuntimeError(f"no match: {pattern!r}")
    return m.group(group)

def js_str(v):
    """Safe JS string literal for scalar text fields."""
    return json.dumps(v, ensure_ascii=False)

def js_body(v):
    """Template-literal body. No backticks/${/backslashes exist in content;
    we only neutralise any </script> sequence as a safeguard."""
    v = re.sub(r'</script', r'<\\/script', v, flags=re.I)
    return "`" + v + "`"

def parse(s):
    icon = first(r'<span class="icon">([^<]*)</span>', s)
    h1 = first(r'<h1>(.*?)</h1>', s)
    hm = re.search(r'^(.*?)<span class="gradient-text">(.*?)</span>\s*$', h1, re.S)
    title = hm.group(1).strip()
    highlight = hm.group(2).strip()
    subtitle = first(r'<p class="subtitle">(.*?)</p>', s).strip()

    stats = []
    for m in re.finditer(r'<div class="stat-item"><div class="stat-num">(.*?)</div><div class="stat-label">(.*?)</div></div>', s, re.S):
        stats.append({'num': m.group(1).strip(), 'label': m.group(2).strip()})

    footer = first(r'<div class="footer">(.*?)</div>\s*<script>', s).strip()

    # Original hand-authored TOC link text per section (may differ from the
    # section heading, e.g. transformer s3). Preserve verbatim.
    toc_map = {}
    toc_ul = re.search(r'<ul class="toc-list">(.*?)</ul>', s, re.S)
    if toc_ul:
        for tm in re.finditer(r'<a href="#(s\d+)">(.*?)</a>', toc_ul.group(1), re.S):
            toc_map[tm.group(1)] = tm.group(2).strip()

    # Walk section + q-card markers in document order.
    markers = []
    for m in re.finditer(r'<div class="section" id="(s\d+)">', s):
        markers.append(('section', m.start(), m.group(1)))
    for m in re.finditer(r'<div class="q-card" id="(q\d+)">', s):
        markers.append(('qcard', m.start(), m.group(1)))
    markers.sort(key=lambda t: t[1])

    sections = []
    cur = None
    for kind, pos, mid in markers:
        if kind == 'section':
            # section-title is the first child; it has no nested divs.
            st = first(r'<div class="section-title">(.*?)</div>', s[pos:], re.S)
            sicon = first(r'<span class="s-icon">([^<]*)</span>', st)
            scount = first(r'<span class="s-count">([^<]*)</span>', st)
            stitle = re.sub(r'<span class="s-icon">[^<]*</span>', '', st)
            stitle = re.sub(r'<span class="s-count">[^<]*</span>', '', stitle).strip()
            cur = {'id': mid, 'icon': sicon, 'title': stitle, 'count': scount,
                   'toc': toc_map.get(mid), 'questions': []}
            sections.append(cur)
        else:  # qcard
            # header region = from q-card start up to q-body open
            body_open_m = re.search(r'<div class="q-body">', s[pos:])
            if not body_open_m:
                raise RuntimeError(f"no q-body for {mid}")
            hdr_end = pos + body_open_m.end()
            header_region = s[pos:pos + body_open_m.start()]
            num = first(r'<span class="q-num">([^<]*)</span>', header_region).strip()
            qtitle = first(r'<h4>(.*?)</h4>', header_region, re.S).strip()
            tags = []
            for tm in re.finditer(r'<span class="q-tag([^"]*)">([^<]*)</span>', header_region):
                cls = tm.group(1).strip()
                tags.append({'text': tm.group(2).strip(), 'cls': cls})
            body = slice_body(s, hdr_end).strip('\n')
            cur['questions'].append({'id': mid, 'num': num, 'title': qtitle, 'tags': tags, 'body': body})

    return {
        'icon': icon, 'title': title, 'highlight': highlight, 'subtitle': subtitle,
        'stats': stats, 'sections': sections, 'footer': footer,
    }

def emit(data, accent, accent_bg, accent_glow):
    lines = []
    lines.append("{")
    lines.append(f"  icon: {js_str(data['icon'])},")
    lines.append(f"  title: {js_str(data['title'])},")
    lines.append(f"  highlight: {js_str(data['highlight'])},")
    lines.append(f"  subtitle: {js_str(data['subtitle'])},")
    lines.append("  stats: [" + ", ".join(
        "{num:" + js_str(st['num']) + ",label:" + js_str(st['label']) + "}" for st in data['stats']
    ) + "],")
    lines.append("  sections: [")
    for sec in data['sections']:
        lines.append("    {")
        toc_field = f", toc: {js_str(sec['toc'])}" if sec.get('toc') else ""
        lines.append(f"      id: {js_str(sec['id'])}, icon: {js_str(sec['icon'])}, title: {js_str(sec['title'])}, count: {js_str(sec['count'])}{toc_field},")
        lines.append("      questions: [")
        for q in sec['questions']:
            tagstr = ", ".join("{text:" + js_str(t['text']) + ",cls:" + js_str(t['cls']) + "}" for t in q['tags'])
            lines.append("        {")
            lines.append(f"          id: {js_str(q['id'])}, num: {js_str(q['num'])},")
            lines.append(f"          title: {js_str(q['title'])},")
            lines.append(f"          tags: [{tagstr}],")
            lines.append(f"          body: {js_body(q['body'])}")
            lines.append("        },")
        lines.append("      ]")
        lines.append("    },")
    lines.append("  ],")
    lines.append(f"  footer: {js_body(data['footer'])}")
    lines.append("}")
    data_js = "\n".join(lines)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{data['title']} 面试题精选</title>
<link rel="stylesheet" href="assets/common.css">
<style>:root{{--accent:{accent};--accent-bg:{accent_bg};--accent-glow:{accent_glow};}}</style>
</head>
<body>
<div id="app"></div>
<script src="assets/interview.js"></script>
<script>
const DATA = {data_js};
initInterview(DATA);
</script>
</body>
</html>
"""

def main():
    import subprocess
    for fname, accent, abg, ag in PAGES:
        path = os.path.join(ROOT, fname)
        # Always regenerate from the original hand-authored source in git HEAD,
        # so the script is idempotent regardless of the current working tree.
        orig = subprocess.check_output(["git", "show", f"HEAD:{fname}"], cwd=ROOT).decode("utf-8")
        data = parse(orig)
        out = emit(data, accent, abg, ag)
        # preserve original <title> text verbatim
        orig_title = first(r'<title>(.*?)</title>', orig)
        out = out.replace(f"<title>{data['title']} 面试题精选</title>", f"<title>{orig_title}</title>", 1)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(out)
        nq = sum(len(s['questions']) for s in data['sections'])
        print(f"{fname}: {len(data['sections'])} sections, {nq} questions -> rewritten")

if __name__ == '__main__':
    main()
