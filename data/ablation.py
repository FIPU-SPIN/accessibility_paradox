import csv, io, re, os, requests, zipfile
import xml.etree.ElementTree as ET

MODEL = "mistral:7b"
PROMPT = ("Explain the following computer science concept accurately to a "
          "student who has reading difficulties. Use simplified language and "
          "short sentences. Do not use analogies.\n\n"
          "Answer in English.\n\nConcept: {c}")
KEYWORDS = ["like a", "imagine", "think of", "as if", "similar to",
            "picture", "just as", "is like", "are like"]

T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
X = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"


def read_ods(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    sheet = next(root.iter("{%s}table" % T))
    rows = []
    for row in sheet.iter("{%s}table-row" % T):
        vals = []
        for cell in row:
            if cell.tag not in ("{%s}table-cell" % T, "{%s}covered-table-cell" % T):
                continue
            rep = int(cell.get("{%s}number-columns-repeated" % T, 1))
            txt = "\n".join("".join(p.itertext()) for p in cell.findall("{%s}p" % X))
            vals.extend([txt] * min(rep, 50))
        while vals and vals[-1] == "":
            vals.pop()
        if vals:
            rrep = int(row.get("{%s}number-rows-repeated" % T, 1))
            rows.extend([vals] * min(rrep, 50))
    header = [h.strip().lower() for h in rows[0]]
    return [dict(zip(header, r)) for r in rows[1:]]


def read_table(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":
        return read_ods(path)
    txt = None
    for enc in ("utf-8-sig", "cp1250", "cp1252"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    if txt is None:
        txt = raw.decode("cp1252", errors="replace")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    first = txt.split("\n")[0]
    delim = ";" if first.count(";") > first.count(",") else ","
    rd = csv.DictReader(io.StringIO(txt), delimiter=delim)
    return [{(k or "").strip().lower(): v for k, v in r.items()} for r in rd]


data = read_table("corpus.csv")
print("Redaka u corpus.csv:", len(data))

rows, seen = [], set()
for r in data:
    c = (r.get("concept") or "").strip()
    if c and c.lower() not in seen:
        seen.add(c.lower())
        rows.append({"ID": r.get("concept_id") or r.get("id") or "", "Concept": c})

print("Jedinstvenih koncepata:", len(rows))
for r in rows:
    print("  ", r["ID"], "|", r["Concept"])

out = []
for r in rows:
    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": MODEL, "stream": False,
        "messages": [{"role": "user", "content": PROMPT.format(c=r["Concept"])}],
        "options": {"temperature": 0.7, "seed": 42, "num_ctx": 16384},
    }, timeout=1800).json()
    text = resp["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    flag = any(k in text.lower() for k in KEYWORDS)
    out.append({"ID": r["ID"], "Concept": r["Concept"],
                "keyword_analogy": flag, "response": text})
    print(len(out), "/", len(rows), r["ID"], "analogy-keyword:", flag)
    with open("ablation_D.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)

print("Gotovo. Spremljeno u ablation_D.csv")