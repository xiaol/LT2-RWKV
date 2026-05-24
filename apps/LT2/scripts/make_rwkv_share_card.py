"""Build the README share-card image for the RWKV-7 vs GDN comparison."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
CURVE_PATH = REPO_ROOT / "apps/LT2/scripts/rwkv7_native_gdn_fineweb_sp1024_param_matched_loss_curve_5000.png"
OUT_PATH = REPO_ROOT / "apps/LT2/scripts/rwkv7_native_gdn_fineweb_sp1024_share_card.png"


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            width = draw.textbbox((0, 0), candidate, font=font)[2]
            if width <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def main() -> None:
    curve = Image.open(CURVE_PATH).convert("RGB")

    width = 1600
    margin = 90
    panel_pad_x = 46
    panel_pad_y = 40
    panel_width = width - 2 * margin
    text_width = panel_width - 2 * panel_pad_x

    font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

    scratch = Image.new("RGB", (width, 400), "#f7f8fb")
    draw = ImageDraw.Draw(scratch)

    headline = "Looped Transformers with linear attention are cool."
    body = (
        "I trained a GDN vs. RWKV-7 comparison in Codex with one command: "
        "\"Give RWKV-7 a try.\" Here is what I got."
    )
    caption = "FineWeb sp1024, parameter-matched tiny LT2 run, bf16 CUDA, 5000 steps"

    headline_lines = wrap_text(draw, headline, font_bold, text_width)
    body_lines = wrap_text(draw, body, font_body, text_width)
    caption_lines = wrap_text(draw, caption, font_small, text_width)

    headline_lh = 62
    body_lh = 49
    caption_lh = 35
    panel_height = (
        2 * panel_pad_y
        + len(headline_lines) * headline_lh
        + 18
        + len(body_lines) * body_lh
        + 20
        + len(caption_lines) * caption_lh
    )

    curve_width = width - 2 * margin
    curve_height = int(curve.height * curve_width / curve.width)
    gap = 34
    height = margin // 2 + panel_height + gap + curve_height + 70

    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)

    panel = [margin, margin // 2, width - margin, margin // 2 + panel_height]
    draw.rounded_rectangle(panel, radius=26, fill="#ffffff", outline="#d8e0ee", width=2)

    x = margin + panel_pad_x
    y = margin // 2 + panel_pad_y
    for line in headline_lines:
        draw.text((x, y), line, font=font_bold, fill="#151922")
        y += headline_lh

    y += 18
    for line in body_lines:
        draw.text((x, y), line, font=font_body, fill="#2f3745")
        y += body_lh

    y += 20
    for line in caption_lines:
        draw.text((x, y), line, font=font_small, fill="#687386")
        y += caption_lh

    resized_curve = curve.resize((curve_width, curve_height), Image.Resampling.LANCZOS)
    img.paste(resized_curve, (margin, panel[3] + gap))
    img.save(OUT_PATH, quality=95)
    print(f"Wrote {OUT_PATH} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
