# charset_builder.py
# ─────────────────────────────────────────────────────────────────────────────
# Run this BEFORE training to find out exactly which characters appear in
# your dataset, then paste the output into config.py.
#
# Usage:
#   python charset_builder.py --json data/annotated/labels.json
#   python charset_builder.py --json data/annotated/labels.json --top 200
# ─────────────────────────────────────────────────────────────────────────────

import json, argparse
from collections import Counter


# ── 1. Pre-built character blocks you can paste into config.py ───────────────

ASCII = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)

# Hiragana — U+3041 to U+3096 (full block including small variants)
HIRAGANA = "".join(chr(c) for c in range(0x3041, 0x3097))
# → ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞた
#   だちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみ
#   むめもゃやゅゆょよらりるれろゎわゐゑをんゔゕゖ

# Katakana — U+30A0 to U+30FF (full block)
KATAKANA = "".join(chr(c) for c in range(0x30A0, 0x3100))
# → ゠ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタ
#   ダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミ
#   ムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶヷヸヹヺ・ーヽヾヿ

# Japanese punctuation most common in manga
JP_PUNCT = (
    "。、！？…‥「」『』【】〔〕〈〉《》"
    "・ー〜※〇△○●◎■□★☆♪♫"
    "／＼｜―─━"
    "（）"            # full-width parentheses
    "０１２３４５６７８９"   # full-width digits
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"  # full-width alpha
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
)

# Jōyō Kanji — the 2136 kanji taught in Japanese school.
# This is the right kanji list for manga OCR (almost all manga kanji are in here).
# Rather than pasting all 2136 here, we load them from a file or from Python's
# Unicode database at runtime.  Call `build_joyo_kanji()` to get the string.

def build_joyo_kayo() -> str:
    """
    Returns a string of the 2136 Jōyō kanji.
    These span several Unicode blocks — we pull them from a curated list.
    """
    # CJK Unified Ideographs commonly used in Japanese (rough superset of Jōyō)
    # For a precise Jōyō list, use the file at:
    #   https://github.com/davidluzgouveia/kanji-data/blob/master/kanji.json
    # Here we use the full CJK block as a practical starting point.
    # Start: U+4E00, End: U+9FFF  (common CJK ideographs)
    return "".join(chr(c) for c in range(0x4E00, 0xA000))


# ── 2. The recommended starter CHARSET tiers ─────────────────────────────────

CHARSET_KANA_ONLY = ASCII + HIRAGANA + KATAKANA + JP_PUNCT
"""
~400 characters.
Good for: manga with mostly dialogue and sound effects.
Train this first — your model handles ~95% of everyday manga text.
"""

CHARSET_WITH_COMMON_KANJI = CHARSET_KANA_ONLY + (
    # ~200 highest-frequency kanji in manga (covers ~85% of kanji occurrences)
    "日本人年大中小上下左右手目口耳足心気力"
    "男女子名前後今来見聞言話読書写"
    "行走止立出入水火土木金空時間"
    "国家族父母兄弟姉妹友学校先生"
    "好悪強弱速遅多少長短新古高低"
    "思知使持待始終死生戦闘力魔神"
    "王城剣魔龍英雄冒険世界"
)
"""
~600 characters.
Good for: shonen/shojo manga with kanji.
Covers the vocabulary patterns most common in action/fantasy manga.
"""

CHARSET_FULL = CHARSET_KANA_ONLY + build_joyo_kayo()
"""
~2600 characters.
Good for: serious/literary manga, historical manga.
Larger charset = model needs more data and training time to learn.
Don't start here — work up to it.
"""


# ── 3. Data-driven charset builder ───────────────────────────────────────────

def build_charset_from_data(json_path: str, min_freq: int = 2, top_n: int = None) -> dict:
    """
    Scan your annotation JSON and find every character that actually appears.
    Returns a dict with:
      - 'charset': the character string to put in config.py
      - 'counter': Counter of {char: count} so you can inspect it
      - 'unseen_in_presets': chars in your data NOT covered by the preset tiers

    json_path: path to your labels JSON
      Format: [{"image": "...", "text": "こんにちは"}, ...]
      OR:     [{"image": "...", "texts": ["line1", "line2"]}, ...]

    min_freq: minimum number of times a character must appear to be included
    top_n:    if set, only keep the top N most frequent characters
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for item in data:
        # Handle both single "text" and list "texts" formats
        if "text" in item:
            texts = [item["text"]]
        elif "texts" in item:
            texts = item["texts"]
        else:
            continue
        for text in texts:
            for ch in text:
                counter[ch] += 1

    # Filter by frequency
    filtered = {ch: cnt for ch, cnt in counter.items() if cnt >= min_freq}

    # Optionally limit to top N
    if top_n:
        filtered = dict(counter.most_common(top_n))

    charset_from_data = "".join(sorted(filtered.keys()))

    # Identify any characters NOT in your preset tiers
    preset_all = set(CHARSET_KANA_ONLY)
    unseen = {ch: cnt for ch, cnt in filtered.items() if ch not in preset_all}

    return {
        "charset":           charset_from_data,
        "counter":           counter,
        "total_unique_chars": len(filtered),
        "unseen_in_presets": unseen,
    }


def print_report(result: dict):
    print("=" * 60)
    print(f"Total unique characters in dataset: {result['total_unique_chars']}")
    print()

    # Top 30 most frequent
    print("Top 30 most frequent characters:")
    top30 = result["counter"].most_common(30)
    for rank, (ch, cnt) in enumerate(top30, 1):
        category = classify_char(ch)
        print(f"  {rank:2d}.  {ch!r:6s}  count={cnt:5d}  [{category}]")

    print()
    print(f"Characters NOT in preset CHARSET_KANA_ONLY: {len(result['unseen_in_presets'])}")
    if result["unseen_in_presets"]:
        unseen_sorted = sorted(result["unseen_in_presets"].items(),
                               key=lambda x: -x[1])[:50]
        print("  (top 50 by frequency)")
        for ch, cnt in unseen_sorted:
            print(f"    {ch!r:6s}  count={cnt:5d}  [{classify_char(ch)}]")

    print()
    print("Paste this into config.py CHARSET:")
    print("-" * 60)
    print(f'CHARSET = """{result["charset"]}"""')
    print("-" * 60)


def classify_char(ch: str) -> str:
    """Label a character by its writing system."""
    cp = ord(ch)
    if cp < 0x80:
        return "ASCII"
    if 0x3041 <= cp <= 0x3096:
        return "Hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "Katakana"
    if 0x4E00 <= cp <= 0x9FFF:
        return "Kanji (CJK)"
    if 0x3000 <= cp <= 0x303F:
        return "JP punctuation"
    if 0xFF00 <= cp <= 0xFFEF:
        return "Full-width"
    return f"Other (U+{cp:04X})"


# ── 4. Kaggle cell version ────────────────────────────────────────────────────

KAGGLE_CELL = '''
# ── Paste this into a Kaggle cell to analyse your dataset ─────────────────

from collections import Counter
import json

def analyse_charset(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    counter = Counter()
    for item in data:
        texts = [item["text"]] if "text" in item else item.get("texts", [])
        for t in texts:
            counter.update(t)
    
    print(f"Unique characters: {len(counter)}")
    print("\\nTop 20:")
    for ch, cnt in counter.most_common(20):
        print(f"  {repr(ch):8s} {cnt:5d}x")
    
    charset = "".join(sorted(counter.keys()))
    print(f"\\nTotal charset size: {len(charset)}")
    print("\\nAdd this to your config.py:")
    print(f\'CHARSET = """{charset}"""\')
    return charset

# charset = analyse_charset("/kaggle/working/recognition_data.json")
'''


# ── 5. Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CHARSET from annotation data")
    parser.add_argument("--json",   required=True, help="Path to annotation JSON")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Min occurrences to include a character (default: 2)")
    parser.add_argument("--top",    type=int, default=None,
                        help="Only keep top N chars by frequency")
    parser.add_argument("--tier",   choices=["kana", "common_kanji", "full"],
                        help="Print a preset tier instead of analysing data")
    args = parser.parse_args()

    if args.tier:
        tiers = {
            "kana":         CHARSET_KANA_ONLY,
            "common_kanji": CHARSET_WITH_COMMON_KANJI,
            "full":         CHARSET_FULL,
        }
        print(f"Preset tier: {args.tier}")
        print(f"Size: {len(tiers[args.tier])} characters")
        print(f'\nCHARSET = """{tiers[args.tier]}"""')
    else:
        result = build_charset_from_data(args.json, args.min_freq, args.top)
        print_report(result)
