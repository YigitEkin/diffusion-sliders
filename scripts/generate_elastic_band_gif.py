from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "generated" / "elastic_band_demo.gif"
WIDTH = 1120
HEIGHT = 620
FPS = 12

BG = (247, 243, 235)
PANEL = (252, 249, 243)
PANEL_ALT = (255, 252, 246)
BORDER = (228, 220, 208)
TEXT = (28, 39, 53)
MUTED = (92, 105, 121)
AXIS = (185, 192, 202)
DASH = (214, 207, 196)
BASE = (96, 119, 146)
HILITE = (181, 134, 58)
MOVE = (78, 159, 127)
NEG = (132, 153, 178)
FADE = (205, 209, 214)
WINDOW = (224, 170, 79)
WHITE = (255, 255, 255)

GEORGIA = "/System/Library/Fonts/Supplemental/Georgia.ttf"
HELVETICA = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


TITLE_FONT = font(GEORGIA, 34)
BODY_FONT = font(HELVETICA, 23)
BODY_BOLD = font(HELVETICA, 24)
SMALL_FONT = font(HELVETICA, 18)
PILL_FONT = font(HELVETICA, 18)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def ease(t: float) -> float:
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def mix(color_a, color_b, t: float):
    return tuple(int(lerp(a, b, t)) for a, b in zip(color_a, color_b))


def smoothstep(frame: int, start: int, duration: int) -> float:
    return ease((frame - start) / float(duration))


def rr(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_text_center(draw: ImageDraw.ImageDraw, xy, text: str, font_obj, fill):
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font_obj, fill=fill)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width=4):
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx**2 + dy**2) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    left = (tip[0] - 18 * ux + 8 * px, tip[1] - 18 * uy + 8 * py)
    right = (tip[0] - 18 * ux - 8 * px, tip[1] - 18 * uy - 8 * py)
    draw.polygon([tip, left, right], fill=color)


def point_x(value: float, left: int, right: int) -> float:
    return lerp(left, right, value / 10.0)


def line_values(points: list[float]) -> list[float]:
    return [round((points[i + 1] - points[i]) / 2.2, 2) for i in range(len(points) - 1)]


def draw_gap(draw: ImageDraw.ImageDraw, left_x: float, right_x: float, y: float, color, label: str, highlight=False):
    pad = 12
    rr(
        draw,
        (left_x + pad, y - 13, right_x - pad, y + 13),
        12,
        mix(WHITE, color, 0.12 if highlight else 0.06),
        color,
        2,
    )
    draw.line((left_x + pad + 8, y, right_x - pad - 8, y), fill=color, width=3)
    draw_text_center(draw, ((left_x + right_x) / 2, y - 24), label, SMALL_FONT, color)


def draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, fill, active=False, faded=False):
    radius = 16 if active else 14
    point_fill = fill if not faded else mix(fill, FADE, 0.75)
    outline = WHITE if not faded else mix(WHITE, FADE, 0.6)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=point_fill, outline=outline, width=4)


def draw_card(draw: ImageDraw.ImageDraw):
    rr(draw, (18, 18, WIDTH - 18, HEIGHT - 18), 34, PANEL, BORDER, 2)
    rr(draw, (42, 44, WIDTH - 42, HEIGHT - 42), 26, PANEL_ALT, BORDER, 2)


def layout_text(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, text: str, font_obj, fill, line_gap=10):
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font_obj)
        if bbox[2] - bbox[0] <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    cursor_y = y
    for line in lines:
        draw.text((x, cursor_y), line, font=font_obj, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font_obj)
        cursor_y += bbox[3] - bbox[1] + line_gap


def stage_text(frame: int) -> tuple[str, str]:
    if frame < 12:
        return (
            "Initialize",
            "Start from a small set of steering values sampled across the current search interval.",
        )
    if frame < 24:
        return (
            "Measure Gaps",
            "Generate outputs at neighboring points and normalize each perceptual distance by the target gap.",
        )
    if frame < 38:
        return (
            "EXPAND",
            "The largest normalized gap exceeds the threshold, so the algorithm inserts a midpoint there.",
        )
    if frame < 52:
        return (
            "MOVE RIGHT",
            "An interior point shifts right because the gap on its right is larger than the gap on its left.",
        )
    if frame < 66:
        return (
            "MOVE LEFT",
            "Another interior point shifts left because its left gap is larger, pulling the band back into balance.",
        )
    if frame < 80:
        return (
            "Converge",
            "Once neighboring gaps are close to uniform, the band stabilizes and the control points stop moving.",
        )
    return (
        "Filter Valid Range",
        "Discard strengths that are too close to the source or too far from it, then keep the usable interior band.",
    )


def render_frame(frame: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw_card(draw)

    left = 100
    right = WIDTH - 100
    top = 118
    axis_y = 422

    title, detail = stage_text(frame)
    title_box = draw.textbbox((0, 0), title, font=TITLE_FONT)
    pill_width = title_box[2] - title_box[0] + 40
    rr(draw, (70, 70, 70 + pill_width, 112), 21, mix(WHITE, WINDOW, 0.16), mix(WINDOW, BORDER, 0.2), 2)
    draw.text((90, 79), title, font=TITLE_FONT, fill=TEXT)
    layout_text(draw, 70, 126, WIDTH - 140, detail, BODY_FONT, MUTED, line_gap=8)

    rr(draw, (66, 208, WIDTH - 66, HEIGHT - 66), 24, mix(WHITE, BASE, 0.03), mix(BORDER, AXIS, 0.35), 2)

    draw.line((left, axis_y, right, axis_y), fill=AXIS, width=4)
    for tick in range(6):
        tx = lerp(left, right, tick / 5)
        draw.line((tx, axis_y - 10, tx, axis_y + 10), fill=AXIS, width=3)

    draw.text((left, axis_y + 18), "low steering", font=SMALL_FONT, fill=MUTED)
    max_box = draw.textbbox((0, 0), "high steering", font=SMALL_FONT)
    draw.text((right - (max_box[2] - max_box[0]), axis_y + 18), "high steering", font=SMALL_FONT, fill=MUTED)

    t_expand = smoothstep(frame, 26, 8)
    t_move_right = smoothstep(frame, 40, 8)
    t_move_left = smoothstep(frame, 54, 8)
    t_filter = smoothstep(frame, 82, 8)

    p0 = [0.6, 2.1, 7.3, 9.4]
    p1 = [0.6, 2.1, 4.7, 7.3, 9.4]
    p2 = [0.6, 2.8, 4.7, 7.3, 9.4]
    p3 = [0.6, 2.8, 5.0, 7.3, 9.4]
    p4 = [0.6, 2.8, 5.0, 7.2, 9.4]

    points = [p0[0], p0[1]]
    inserted = lerp((p0[1] + p0[2]) / 2, p1[2], t_expand)
    third = lerp(p0[2], p1[3], t_expand)
    fourth = p1[4]
    if t_expand < 1:
        points.extend([inserted, third, fourth])
    else:
        points = p1[:]

    if t_expand >= 1:
        points[1] = lerp(p1[1], p2[1], t_move_right)
        points[2] = lerp(p1[2], p3[2], t_move_right)
        points[3] = lerp(p1[3], p4[3], t_move_left)

    gaps = line_values(points)
    target_start = point_x(0.6, left, right)
    target_end = point_x(9.4, left, right)
    uniform = [0.6, 2.8, 5.0, 7.2, 9.4]

    draw.line((target_start, axis_y - 118, target_end, axis_y - 118), fill=DASH, width=2)
    for anchor in uniform:
        x = point_x(anchor, left, right)
        draw.line((x, axis_y - 128, x, axis_y - 108), fill=DASH, width=2)

    overlay = "Target equalized spacing"
    overlay_box = draw.textbbox((0, 0), overlay, font=PILL_FONT)
    rr(
        draw,
        (
            WIDTH - 90 - (overlay_box[2] - overlay_box[0]) - 28,
            218,
            WIDTH - 90,
            252,
        ),
        16,
        mix(WHITE, DASH, 0.18),
        mix(DASH, BORDER, 0.35),
        2,
    )
    draw.text((WIDTH - 104 - (overlay_box[2] - overlay_box[0]), 226), overlay, font=PILL_FONT, fill=MUTED)

    highlight_gap = None
    if 12 <= frame < 38:
        gap_idx = max(range(len(gaps)), key=lambda idx: gaps[idx])
        highlight_gap = gap_idx

    point_xs = [point_x(value, left, right) for value in points]
    gap_y = axis_y - 68
    for i, gap in enumerate(gaps):
        color = HILITE if i == highlight_gap else BASE
        draw_gap(draw, point_xs[i], point_xs[i + 1], gap_y - 22 * (i % 2), color, f"G{i + 1}={gap:.2f}", i == highlight_gap)

    if 12 <= frame < 24:
        hint = "largest"
        hx = (point_xs[1] + point_xs[2]) / 2
        hint_box = draw.textbbox((0, 0), hint, font=SMALL_FONT)
        rr(draw, (hx - 38, axis_y - 170, hx + 38, axis_y - 138), 12, mix(WHITE, HILITE, 0.18), mix(HILITE, BORDER, 0.2), 2)
        draw_text_center(draw, (hx, axis_y - 154), hint, SMALL_FONT, HILITE)

    if 26 <= frame < 38:
        mid_x = point_xs[2]
        pulse_r = 18 + (frame % 4) * 4
        draw.ellipse((mid_x - pulse_r, axis_y - pulse_r, mid_x + pulse_r, axis_y + pulse_r), outline=HILITE, width=3)

    active_idx = None
    if 40 <= frame < 52:
        active_idx = 1
    elif 52 <= frame < 66:
        active_idx = 3

    for idx, x in enumerate(point_xs):
        faded = t_filter > 0.1 and idx in (0, len(point_xs) - 1)
        color = MOVE if idx == active_idx else BASE
        if 80 <= frame:
            color = MOVE
        draw_point(draw, x, axis_y, color, active=idx == active_idx, faded=faded)

    if 40 <= frame < 52:
        current = point_xs[1]
        start = point_x(p1[1], left, right)
        end = point_x(p2[1], left, right)
        draw_arrow(draw, (start, axis_y - 48), (end, axis_y - 48), MOVE, width=4)
        draw_text_center(draw, ((start + end) / 2, axis_y - 76), "shift toward larger right gap", SMALL_FONT, MOVE)
    if 52 <= frame < 66:
        start = point_x(p3[3], left, right)
        end = point_x(p4[3], left, right)
        draw_arrow(draw, (start, axis_y - 48), (end, axis_y - 48), MOVE, width=4)
        draw_text_center(draw, ((start + end) / 2, axis_y - 76), "shift back toward larger left gap", SMALL_FONT, MOVE)

    if 66 <= frame < 80:
        span_left = point_x(uniform[0], left, right)
        span_right = point_x(uniform[-1], left, right)
        rr(draw, (span_left - 10, axis_y - 32, span_right + 10, axis_y + 32), 18, mix(WHITE, MOVE, 0.06), mix(MOVE, BORDER, 0.25), 2)
        draw_text_center(draw, ((span_left + span_right) / 2, axis_y - 96), "neighboring gaps are now near the target gap", SMALL_FONT, MOVE)

    if t_filter > 0:
        valid_left = point_x(2.8, left, right)
        valid_right = point_x(7.2, left, right)
        draw.line((valid_left, axis_y + 70, valid_right, axis_y + 70), fill=WINDOW, width=6)
        draw.line((valid_left, axis_y + 54, valid_left, axis_y + 86), fill=WINDOW, width=6)
        draw.line((valid_right, axis_y + 54, valid_right, axis_y + 86), fill=WINDOW, width=6)
        rr(draw, (valid_left - 56, axis_y + 96, valid_right + 56, axis_y + 136), 18, mix(WHITE, WINDOW, 0.12), mix(WINDOW, BORDER, 0.28), 2)
        draw_text_center(draw, ((valid_left + valid_right) / 2, axis_y + 116), "keep the interior steering range", BODY_BOLD, WINDOW)

    return image


def main():
    frames = [render_frame(frame) for frame in range(96)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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
