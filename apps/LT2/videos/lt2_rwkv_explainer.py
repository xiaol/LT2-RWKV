from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from manim import (
    BLACK,
    BLUE_C,
    BLUE_D,
    BOLD,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN_C,
    Group,
    LEFT,
    ORANGE,
    RIGHT,
    Scene,
    Text,
    UP,
    VGroup,
    WHITE,
    Write,
    ImageMobject,
    RoundedRectangle,
    Arrow,
    GrowArrow,
    Create,
    ReplacementTransform,
    TransformFromCopy,
    Circumscribe,
    BarChart,
    config,
)


ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = ROOT / "apps/LT2/scripts"
LOSS_CURVE = ASSET_DIR / "rwkv7_native_gdn_fineweb_sp1024_param_matched_loss_curve_5000.png"
SHARE_CARD = ASSET_DIR / "rwkv7_native_gdn_fineweb_sp1024_share_card.png"

config.background_color = BLACK
config.frame_width = 16
config.frame_height = 9
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = float(Fraction(30000, 1001))
config.media_dir = str(ROOT / "apps/LT2/videos/media")

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
INK = "#f4f0e6"
MUTED = "#9aa4ad"
PANEL = "#111820"
GDN = BLUE_C
RWKV = GREEN_C
ACCENT = ORANGE
PURPLE = "#9b7bff"


class LT2RWKVExplainer(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        self.opening()
        self.looped_transformer()
        self.replacement()
        self.benchmark()
        self.loss_curve()
        self.repo_card()

    def opening(self):
        title = Text(
            "LT2-RWKV",
            font=FONT,
            weight=BOLD,
            color=INK,
            font_size=74,
        )
        subtitle = Text(
            "What if the looped linear-attention mixer is RWKV-7 instead of GDN?",
            font=FONT,
            color=MUTED,
            font_size=31,
        )
        title = self.fit_width(title)
        subtitle = self.fit_width(subtitle)
        subtitle.next_to(title, DOWN, buff=0.22)
        repo = Text(
            "github.com/xiaol/LT2-RWKV",
            font=MONO,
            color=RWKV,
            font_size=25,
        ).next_to(subtitle, DOWN, buff=0.42)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(subtitle, shift=0.25 * DOWN), FadeIn(repo, shift=0.25 * DOWN), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, subtitle, repo), shift=0.25 * UP), run_time=0.7)

    def looped_transformer(self):
        title = self.fit_width(Text("Looping turns one block into many effective steps", font=FONT, weight=BOLD, color=INK, font_size=40))
        title.to_edge(UP, buff=0.38)

        tokens = self.token_row().to_edge(LEFT, buff=0.9).shift(UP * 0.45)
        block = self.block("shared LT2 block", PURPLE, width=3.2).next_to(tokens, RIGHT, buff=1.0)
        out = self.state_box("deeper state", ACCENT).next_to(block, RIGHT, buff=1.0)
        arrow1 = Arrow(tokens.get_right(), block.get_left(), buff=0.16, color=INK)
        arrow2 = Arrow(block.get_right(), out.get_left(), buff=0.16, color=INK)

        passes = VGroup(*[
            Text(f"pass {i}", font=MONO, color=ACCENT, font_size=24)
            for i in range(1, 5)
        ]).arrange(RIGHT, buff=0.35)
        passes.next_to(block, DOWN, buff=0.38)
        loop_arrow = Arrow(
            block.get_bottom() + DOWN * 0.25 + RIGHT * 1.35,
            block.get_bottom() + DOWN * 0.25 + LEFT * 1.35,
            path_arc=-2.8,
            color=PURPLE,
            stroke_width=5,
            buff=0.05,
        )

        note = Text("same parameters, more recurrent computation", font=FONT, color=MUTED, font_size=27)
        note.to_edge(DOWN, buff=0.8)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(tokens, shift=0.25 * RIGHT), Create(block[0]), Write(block[1]), run_time=1.0)
        self.play(GrowArrow(arrow1), GrowArrow(arrow2), FadeIn(out, shift=0.2 * RIGHT), run_time=0.8)
        self.play(Create(loop_arrow), FadeIn(passes, lag_ratio=0.18), run_time=1.2)
        self.play(FadeIn(note, shift=0.2 * UP), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(VGroup(title, tokens, block, out, arrow1, arrow2, passes, loop_arrow, note)), run_time=0.7)

    def replacement(self):
        title = self.fit_width(Text("The experiment: replace GDN with full RWKV-7", font=FONT, weight=BOLD, color=INK, font_size=40))
        title.to_edge(UP, buff=0.38)

        gdn = self.mixer_panel("GDN", "DPLR recurrent mixer", GDN)
        rwkv = self.mixer_panel("RWKV-7 native", "time mix + channel mix", RWKV)
        gdn.shift(LEFT * 3.6)
        rwkv.shift(RIGHT * 3.6)
        arrow = Arrow(gdn.get_right(), rwkv.get_left(), buff=0.3, color=ACCENT, stroke_width=6)
        label = Text("same LT2 loop, new recurrent core", font=FONT, color=INK, font_size=28).next_to(arrow, UP, buff=0.25)

        code = self.code_panel('layer_pattern = "rwkv7_native"').to_edge(DOWN, buff=0.65)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(gdn, shift=0.3 * RIGHT), run_time=0.9)
        self.play(GrowArrow(arrow), FadeIn(label), ReplacementTransform(gdn.copy(), rwkv), run_time=1.2)
        self.play(FadeIn(code, shift=0.25 * UP), Circumscribe(code, color=RWKV), run_time=1.2)
        self.wait(1.6)
        self.play(FadeOut(VGroup(title, gdn, rwkv, arrow, label, code)), run_time=0.7)

    def benchmark(self):
        title = self.fit_width(Text("First real-token signal: FineWeb sp1024", font=FONT, weight=BOLD, color=INK, font_size=40))
        title.to_edge(UP, buff=0.38)

        left = self.metric_box("GDN", "962,832 params", "43,336 tok/s", "val 4.1191", GDN)
        right = self.metric_box("RWKV-7 native", "959,360 params", "67,919 tok/s", "val 3.8693", RWKV)
        VGroup(left, right).arrange(RIGHT, buff=0.75).move_to(UP * 0.35)

        chart = BarChart(
            values=[43336, 67919],
            bar_names=["GDN", "RWKV-7"],
            y_range=[0, 75000, 25000],
            y_length=2.6,
            x_length=4.5,
            bar_colors=[GDN, RWKV],
        )
        chart.scale(0.72).to_edge(DOWN, buff=0.55).shift(LEFT * 3.1)
        chart_title = Text("tokens / second", font=FONT, color=MUTED, font_size=22).next_to(chart, UP, buff=0.1)

        takeaway = Text("~1.57x faster, lower validation loss", font=FONT, weight=BOLD, color=ACCENT, font_size=36)
        takeaway.to_edge(DOWN, buff=0.82).shift(RIGHT * 3.0)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(left, shift=0.25 * RIGHT), FadeIn(right, shift=0.25 * LEFT), run_time=1.0)
        self.play(Create(chart), FadeIn(chart_title), run_time=1.0)
        self.play(FadeIn(takeaway, shift=0.2 * UP), Circumscribe(right, color=RWKV), run_time=1.2)
        self.wait(1.8)
        self.play(FadeOut(VGroup(title, left, right, chart, chart_title, takeaway)), run_time=0.7)

    def loss_curve(self):
        title = self.fit_width(Text("The curve is the important part", font=FONT, weight=BOLD, color=INK, font_size=40))
        title.to_edge(UP, buff=0.32)
        curve = ImageMobject(str(LOSS_CURVE))
        curve.set_width(12.8)
        curve.next_to(title, DOWN, buff=0.3)
        caption = Text(
            "5000 steps, 5.12M train tokens/model, parameter matched within 0.36%",
            font=FONT,
            color=MUTED,
            font_size=25,
        ).to_edge(DOWN, buff=0.35)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(curve, shift=0.25 * UP), run_time=1.0)
        self.play(FadeIn(caption, shift=0.15 * UP), run_time=0.7)
        self.wait(2.4)
        self.play(FadeOut(Group(title, curve, caption)), run_time=0.7)

    def repo_card(self):
        card = ImageMobject(str(SHARE_CARD))
        card.set_width(13.0)
        card.move_to(UP * 0.05)
        footer = Text(
            "github.com/xiaol/LT2-RWKV",
            font=MONO,
            color=RWKV,
            font_size=30,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(card, shift=0.25 * UP), run_time=1.0)
        self.play(FadeIn(footer, shift=0.18 * UP), run_time=0.8)
        self.wait(2.0)

    def token_row(self):
        group = VGroup()
        for i, txt in enumerate(["x0", "x1", "x2", "..."]):
            box = RoundedRectangle(width=0.75, height=0.58, corner_radius=0.12, color=BLUE_D, fill_color="#132336", fill_opacity=0.9)
            label = Text(txt, font=MONO, color=INK, font_size=22).move_to(box)
            group.add(VGroup(box, label))
        group.arrange(RIGHT, buff=0.15)
        return group

    def state_box(self, label, color):
        box = RoundedRectangle(width=2.2, height=0.8, corner_radius=0.14, color=color, fill_color=PANEL, fill_opacity=0.96, stroke_width=3)
        text = Text(label, font=FONT, color=INK, font_size=23).move_to(box)
        return VGroup(box, text)

    def block(self, label, color, width=2.4):
        box = RoundedRectangle(width=width, height=1.25, corner_radius=0.16, color=color, fill_color=PANEL, fill_opacity=0.96, stroke_width=4)
        text = Text(label, font=FONT, color=INK, font_size=25).move_to(box)
        return VGroup(box, text)

    def mixer_panel(self, title, subtitle, color):
        outer = RoundedRectangle(width=4.2, height=3.0, corner_radius=0.18, color=color, fill_color=PANEL, fill_opacity=0.96, stroke_width=4)
        head = Text(title, font=FONT, weight=BOLD, color=color, font_size=32)
        sub = Text(subtitle, font=FONT, color=INK, font_size=22)
        stack = VGroup(head, sub).arrange(DOWN, buff=0.18).move_to(outer)
        layers = VGroup(*[
            RoundedRectangle(width=2.7, height=0.22, corner_radius=0.05, color=color, fill_color=color, fill_opacity=0.45)
            for _ in range(4)
        ]).arrange(DOWN, buff=0.12).next_to(stack, DOWN, buff=0.35)
        return VGroup(outer, stack, layers)

    def code_panel(self, code):
        box = RoundedRectangle(width=7.1, height=0.78, corner_radius=0.12, color="#3a4550", fill_color="#080d12", fill_opacity=0.97)
        text = Text(code, font=MONO, color=RWKV, font_size=27).move_to(box)
        return VGroup(box, text)

    def metric_box(self, title, params, speed, val, color):
        box = RoundedRectangle(width=5.1, height=2.75, corner_radius=0.18, color=color, fill_color=PANEL, fill_opacity=0.97, stroke_width=4)
        lines = VGroup(
            Text(title, font=FONT, weight=BOLD, color=color, font_size=33),
            Text(params, font=MONO, color=INK, font_size=24),
            Text(speed, font=MONO, color=INK, font_size=24),
            Text(val, font=MONO, color=INK, font_size=24),
        ).arrange(DOWN, buff=0.18)
        lines.move_to(box)
        return VGroup(box, lines)

    def fit_width(self, mob, max_width: float = 14.2):
        if mob.width > max_width:
            mob.scale(max_width / mob.width)
        return mob
