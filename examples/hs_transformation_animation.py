from manim import *

class HSTransformation(Scene):
    def construct(self):
        # COLORS
        FERMION_COLOR = BLUE
        BOSON_COLOR = YELLOW
        EQUATION_COLOR = WHITE

        # --- SCENE 1: THE CURSE OF 4-FERMION ---
        title1 = Text("1. The 4-Fermion Interaction", font_size=36).to_edge(UP)
        eq1 = MathTex(
            r"\exp\left( V \int d\tau c^\dagger c^\dagger c c \right)",
            color=EQUATION_COLOR
        ).shift(UP * 1.5)

        # 4-Fermion Vertex Diagram
        center = ORIGIN + DOWN * 1.0
        lines = VGroup(*[
            Line(center + 1.5 * rotate_vector(RIGHT, angle), center, color=FERMION_COLOR)
            for angle in [PI/4, 3*PI/4, 5*PI/4, 7*PI/4]
        ])
        vertex = Dot(center, color=WHITE)
        diagram1 = VGroup(lines, vertex)

        self.play(Write(title1))
        self.play(Write(eq1))
        self.play(Create(diagram1))
        self.wait(2)

        # --- SCENE 2: HS DECOUPLING ---
        title2 = Text("2. Hubbard-Stratonovich Decoupling", font_size=36).to_edge(UP)
        eq2 = MathTex(
            r"\int \mathcal{D}\phi \exp\left( -\frac{\phi^2}{V} + \phi c^\dagger c \right)",
            color=EQUATION_COLOR
        ).shift(UP * 1.5)

        # Splitting the vertex
        left_v = center + LEFT * 1.0
        right_v = center + RIGHT * 1.0
        
        # New lines for split vertex
        new_lines_l = VGroup(
            Line(left_v + 1.2 * rotate_vector(RIGHT, 3*PI/4), left_v, color=FERMION_COLOR),
            Line(left_v + 1.2 * rotate_vector(RIGHT, 5*PI/4), left_v, color=FERMION_COLOR)
        )
        new_lines_r = VGroup(
            Line(right_v + 1.2 * rotate_vector(RIGHT, PI/4), right_v, color=FERMION_COLOR),
            Line(right_v + 1.2 * rotate_vector(RIGHT, 7*PI/4), right_v, color=FERMION_COLOR)
        )
        wavy_phi = DashedLine(left_v, right_v, color=BOSON_COLOR).set_stroke(width=6)
        
        dot_l = Dot(left_v, color=WHITE)
        dot_r = Dot(right_v, color=WHITE)
        diagram2 = VGroup(new_lines_l, new_lines_r, wavy_phi, dot_l, dot_r)

        self.play(Transform(title1, title2), Transform(eq1, eq2))
        self.play(
            ReplacementTransform(lines[1:3], new_lines_l),
            ReplacementTransform(lines[0:4:3], new_lines_r),
            Create(wavy_phi),
            ReplacementTransform(vertex.copy(), dot_l),
            ReplacementTransform(vertex, dot_r)
        )
        self.play(Indicate(wavy_phi))
        phi_label = MathTex(r"\phi", color=BOSON_COLOR).next_to(wavy_phi, UP)
        self.play(Write(phi_label))
        self.wait(2)

        # --- SCENE 3: INTEGRATING OUT FERMIONS ---
        title3 = Text("3. Effective Action (Exact Tr[ln])", font_size=36).to_edge(UP)
        eq3 = MathTex(
            r"S_{eff}[\phi] = -\frac{\phi^2}{V} - \text{Tr}\ln(1 - G\phi)",
            color=EQUATION_COLOR
        ).shift(UP * 1.5)

        # Transformation to Bubble Diagram
        circle_center = center
        radius = 0.8
        fermion_loop = Circle(radius=radius, color=FERMION_COLOR).move_to(circle_center)
        
        # External boson lines (wavy)
        ext_phi1 = DashedLine(circle_center + LEFT * 2.0, circle_center + LEFT * radius, color=BOSON_COLOR)
        ext_phi2 = DashedLine(circle_center + RIGHT * 2.0, circle_center + RIGHT * radius, color=BOSON_COLOR)
        
        diagram3 = VGroup(fermion_loop, ext_phi1, ext_phi2)

        self.play(Transform(title1, title3), Transform(eq1, eq3))
        self.play(FadeOut(phi_label))
        self.play(
            ReplacementTransform(VGroup(new_lines_l, new_lines_r), fermion_loop),
            ReplacementTransform(wavy_phi, VGroup(ext_phi1, ext_phi2)),
            FadeOut(dot_l), FadeOut(dot_r)
        )
        
        self.play(Circumscribe(fermion_loop))
        tr_label = Text("Fermion Loop", font_size=24, color=FERMION_COLOR).next_to(fermion_loop, DOWN)
        self.play(Write(tr_label))
        self.wait(3)

        # Final Fade
        self.play(FadeOut(VGroup(title1, eq1, diagram3, tr_label)))
        thanks = Text("Physics Visualization by Eliashberg.jl", font_size=32, gradient=(BLUE, YELLOW))
        self.play(Write(thanks))
        self.wait(2)
