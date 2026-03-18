from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "generated" / "steering_vector_demo.gif"
WIDTH = 900
HEIGHT = 500
FPS = 14
TOTAL_FRAMES = 42


def load_font(size: int, *, italic: bool = False, mono: bool = False):
    candidates = []
    if mono:
        candidates = [
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Courier New.ttf",
        ]
    elif italic:
        candidates = [
            "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
            "/Library/Fonts/Georgia Italic.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Georgia.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]

    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FONT_SMALL = load_font(14)
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease(t: float) -> float:
    return 1 - (1 - t) ** 3


def blend(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def draw_round_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text(draw, xy, text, font, fill, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_panel(draw, box):
    draw_round_rect(draw, box, 26, fill=(255, 252, 246), outline=(220, 212, 198), width=2)


def draw_clusters(draw, frame_idx):
    t_points = ease(min(frame_idx / 14, 1))
    t_means = ease(max(0, min((frame_idx - 10) / 12, 1)))
    t_arrow = ease(max(0, min((frame_idx - 20) / 14, 1)))

    chart = (34, 34, 866, 466)
    draw_panel(draw, chart)

    # soft chart background
    draw_round_rect(draw, (70, 70, 830, 440), 20, fill=(248, 249, 252), outline=(229, 234, 240), width=1)
    draw.line((122, 392, 782, 392), fill=(196, 205, 216), width=2)
    draw.line((122, 392, 122, 110), fill=(196, 205, 216), width=2)

    neg_pts = [(258, 338), (312, 276), (354, 354)]
    pos_pts = [(534, 228), (606, 188), (652, 258)]
    neg_mean = (308, 324)
    pos_mean = (597, 225)

    # clusters
    neg_fill = blend((255, 255, 255), (229, 175, 158), 0.35 * t_points)
    pos_fill = blend((255, 255, 255), (168, 215, 195), 0.42 * t_points)
    draw.ellipse((neg_mean[0] - 102, neg_mean[1] - 72, neg_mean[0] + 102, neg_mean[1] + 72), fill=neg_fill, outline=(218, 176, 163))
    draw.ellipse((pos_mean[0] - 114, pos_mean[1] - 80, pos_mean[0] + 114, pos_mean[1] + 80), fill=pos_fill, outline=(169, 204, 188))

    for x, y in neg_pts:
        r = 10
        fill = blend((255, 255, 255), (199, 112, 83), t_points)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255), width=3)
    for x, y in pos_pts:
        r = 10
        fill = blend((255, 255, 255), (78, 159, 127), t_points)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(255, 255, 255), width=3)

    for mean, color in [(neg_mean, (199, 112, 83)), (pos_mean, (78, 159, 127))]:
        halo = 18 + int(6 * math.sin(frame_idx / 4))
        if t_means > 0:
            r_outer = int(lerp(4, 17, t_means))
            r_inner = int(lerp(2, 11, t_means))
            draw.ellipse((mean[0] - halo, mean[1] - halo, mean[0] + halo, mean[1] + halo), outline=color + (0,), width=0)
            draw.ellipse((mean[0] - r_outer, mean[1] - r_outer, mean[0] + r_outer, mean[1] + r_outer), outline=color, width=3)
            draw.ellipse((mean[0] - r_inner, mean[1] - r_inner, mean[0] + r_inner, mean[1] + r_inner), fill=color, outline=(255, 255, 255), width=3)

    # arrow
    end_x = lerp(neg_mean[0], pos_mean[0], t_arrow)
    end_y = lerp(neg_mean[1], pos_mean[1], t_arrow)
    draw.line((neg_mean[0], neg_mean[1], end_x, end_y), fill=(141, 106, 47), width=5)
    if t_arrow > 0.1:
        angle = math.atan2(pos_mean[1] - neg_mean[1], pos_mean[0] - neg_mean[0])
        size = 14
        tip = (end_x, end_y)
        left = (end_x - size * math.cos(angle) + size * 0.5 * math.sin(angle),
                end_y - size * math.sin(angle) - size * 0.5 * math.cos(angle))
        right = (end_x - size * math.cos(angle) - size * 0.5 * math.sin(angle),
                 end_y - size * math.sin(angle) + size * 0.5 * math.cos(angle))
        draw.polygon([tip, left, right], fill=(141, 106, 47))

    # labels
    if t_means > 0:
        for x, y, fill, outline in [
            (248, 362, (249, 239, 234), (215, 186, 176)),
            (554, 154, (235, 246, 240), (178, 211, 196)),
        ]:
            draw_round_rect(draw, (x, y, x + 24, y + 24), 12, fill=fill, outline=outline, width=1)

    if t_arrow > 0:
        draw_round_rect(draw, (424, 252, 448, 276), 12, fill=(248, 241, 227), outline=(214, 192, 150), width=1)


def render_frame(frame_idx: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (247, 243, 235))
    draw = ImageDraw.Draw(img)

    # background
    draw_round_rect(draw, (16, 16, WIDTH - 16, HEIGHT - 16), 32, fill=(252, 249, 243), outline=(229, 220, 208), width=2)

    draw_clusters(draw, frame_idx)
    return img


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_frame(i) for i in range(TOTAL_FRAMES)]
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=False,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
