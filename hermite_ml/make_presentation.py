"""
Assemble the study's PDF presentation (16:9 slides) from already-saved PNGs
plus a few auxiliary schematic figures (see make_feature_figures.py, run it
first if hermite_ml/plots/feature_*.png are missing).

One page per slide; each page is a plain matplotlib Figure written into a
single PDF via PdfPages. No LaTeX/browser dependency.

Usage
-----
python3 hermite_ml/make_feature_figures.py   # if not already run
python3 hermite_ml/make_presentation.py
"""
import os
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.join(os.path.dirname(__file__), '..')
PLOTS = os.path.join(os.path.dirname(__file__), 'plots')
RECON_PLOTS = os.path.join(ROOT, 'reconnection_2d_beta025', 'plots')
OUT_PDF = os.path.join(os.path.dirname(__file__), 'hermite_corrector_presentation.pdf')

PAGE_W, PAGE_H = 13.333, 7.5   # 16:9

NAVY = '#1B2A4A'
ACCENT = '#D8543B'
GRAY = '#555555'


def new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    return fig


def add_title_bar(fig, title, subtitle=None):
    fig.text(0.045, 0.945, title, fontsize=22, fontweight='bold', color=NAVY,
             ha='left', va='top')
    if subtitle:
        fig.text(0.045, 0.895, subtitle, fontsize=12.5, color=GRAY,
                 ha='left', va='top')
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.875, 0.875],
                              transform=fig.transFigure, color=NAVY, lw=1.2))


def add_image(fig, path, rect, caption=None):
    """rect = [left, bottom, width, height] in figure fraction."""
    ax = fig.add_axes(rect)
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis('off')
    if caption:
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha='center',
               va='top', fontsize=9.5, color=GRAY)
    return ax


def add_bullets(fig, rect, bullets, fontsize=13, title=None):
    """
    Renders `bullets` (each a top-level '•' line, or a sub-'–' line if it
    starts with two spaces) into `rect`, wrapping text to fit the rect width
    and scaling line spacing so the whole block always fits the rect height
    -- avoids overflow regardless of how much text a slide has.
    """
    left, bottom, width, height = rect
    ax = fig.add_axes(rect)
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig_w_in = fig.get_size_inches()[0]
    rect_w_in = width * fig_w_in
    chars_per_line_top = max(20, int(rect_w_in * 72 / fontsize * 1.9))
    chars_per_line_sub = max(18, int(chars_per_line_top * 0.92))

    blocks = []
    for b in bullets:
        indent = b.startswith('  ')
        text = b.strip()
        wrap_w = chars_per_line_sub if indent else chars_per_line_top
        lines = textwrap.wrap(text, width=wrap_w) or ['']
        blocks.append((indent, lines))

    total_lines = sum(len(lines) for _, lines in blocks)
    gap_units = len(blocks) * 0.35
    title_units = 1.8 if title else 0.0
    lh = 1.0 / max(total_lines + gap_units + title_units, 1.0)

    y = 1.0
    if title:
        ax.text(0.0, y, title, fontsize=fontsize + 1, fontweight='bold', color=NAVY,
               ha='left', va='top', transform=ax.transAxes)
        y -= lh * 1.8
    for indent, lines in blocks:
        bullet_char = '–' if indent else '•'
        x0 = 0.07 if indent else 0.0
        x_cont = x0 + 0.035
        for i, line in enumerate(lines):
            prefix = f'{bullet_char} ' if i == 0 else ''
            ax.text(x0 if i == 0 else x_cont, y, f'{prefix}{line}',
                   fontsize=fontsize, color='#222222', ha='left', va='top',
                   transform=ax.transAxes)
            y -= lh
        y -= lh * 0.35
    return ax


def title_slide(pdf):
    fig = new_page()
    fig.patch.set_facecolor(NAVY)
    fig.text(0.5, 0.60, 'An MLP-Based Safeguard for\nHermite-Basis VDF Reconstruction',
             fontsize=28, fontweight='bold', color='white', ha='center', va='center')
    fig.text(0.5, 0.40, 'Removing Gibbs-type artifacts from truncated Hermite\n'
                        'decompositions of strongly non-Maxwellian velocity\n'
                        'distribution functions in a magnetic reconnection simulation',
             fontsize=14.5, color='#CBD5E8', ha='center', va='center')
    fig.text(0.5, 0.20, 'Vlasiator 2D GEM-challenge reconnection run — reconnection_2d_beta025',
             fontsize=11.5, color='#8FA3C8', ha='center', va='center', style='italic')
    pdf.savefig(fig)
    plt.close(fig)


def agenda_slide(pdf):
    fig = new_page()
    add_title_bar(fig, 'Outline')
    bullets = [
        '1. The simulation: 2D magnetic reconnection (GEM challenge)',
        '2. Velocity distribution functions in the current layer',
        '3. Hermite-basis reconstruction and its Gibbs-artifact problem',
        '4. ML corrector design: f_final = f_rec * exp(Δ_pred)',
        '5. Feature evolution: low-S coefficients → 1-D spectra → local patch',
        '6. Within-snapshot results',
        '7. Cross-timestep generalization: failure and fix',
        '8. Conclusions and open weaknesses',
    ]
    add_bullets(fig, [0.08, 0.10, 0.85, 0.7], bullets, fontsize=16)
    pdf.savefig(fig)
    plt.close(fig)


def reconnection_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '1. The Simulation: 2D Magnetic Reconnection',
                 'GEM-challenge current sheet, β=0.25 — Vx colormap with B field lines, '
                 'ion density profile, and a sample VDF at (1000 km, 0, 0)')
    add_image(fig, os.path.join(RECON_PLOTS, 'plot_Vx_00022.png'),
             [0.03, 0.06, 0.46, 0.75], caption='Early time (bulk.0000022) — thin, still-forming current sheet')
    add_image(fig, os.path.join(RECON_PLOTS, 'plot_Vx_00111.png'),
             [0.51, 0.06, 0.46, 0.75], caption='Late time (bulk.0000111) — fully developed reconnection outflows')
    pdf.savefig(fig)
    plt.close(fig)


def vdf_examples_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '2. VDFs in the Current Layer',
                 'Two representative cells from bulk.0000024: a lobe cell (near-Maxwellian) '
                 'vs. the current-sheet center (strongly non-Maxwellian)')
    add_image(fig, os.path.join(PLOTS, 'vdf_comparison_16.png'),
             [0.02, 0.06, 0.47, 0.78], caption='cid=16 — lobe, far from the current sheet')
    add_image(fig, os.path.join(PLOTS, 'vdf_comparison_672.png'),
             [0.50, 0.06, 0.47, 0.78], caption='cid=672 — current-sheet center')
    pdf.savefig(fig)
    plt.close(fig)


def vdf_cuts_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '2. VDFs in the Current Layer (cont.)',
                 '1-D cuts through the integrated VDF, original vs. Hermite reconstruction '
                 'at increasing order — note the ringing that appears for the non-Maxwellian cell')
    add_image(fig, os.path.join(PLOTS, 'vdf_1d_cuts_16.png'),
             [0.02, 0.06, 0.47, 0.78], caption='cid=16 (lobe) — converges cleanly')
    add_image(fig, os.path.join(PLOTS, 'vdf_1d_cuts_672.png'),
             [0.50, 0.06, 0.47, 0.78], caption='cid=672 (current sheet) — Gibbs oscillations at low order')
    pdf.savefig(fig)
    plt.close(fig)


def problem_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '3. The Gibbs-Artifact Problem',
                 'Residuals and Hermite-space power spectra for the current-sheet cell (cid=672)')
    add_image(fig, os.path.join(PLOTS, 'vdf_residuals_672.png'),
             [0.02, 0.06, 0.47, 0.78], caption='|reconstruction − truth|: ringing concentrated at steep gradients')
    add_image(fig, os.path.join(PLOTS, 'spectra_672.png'),
             [0.50, 0.06, 0.47, 0.78], caption='Hermite spectrum: slow decay → truncation always leaves artifacts')
    pdf.savefig(fig)
    plt.close(fig)


def problem_slide2(pdf):
    fig = new_page()
    add_title_bar(fig, '3. The Gibbs-Artifact Problem (cont.)',
                 'Even at high order, 20–38% of in-support voxels go negative in log-space; '
                 'more orders alone do not remove the ringing')
    add_bullets(fig, [0.05, 0.10, 0.42, 0.70], [
        'Truncated tetrahedral Hermite decomposition (up to order 22) acts as a ~60³ velocity-space → ~2300-coefficient compression',
        'Sign-flip (Gibbs) ringing appears near steep gradients / turning points',
        'sparse_mask only zeroes the true-empty region — it does not fix internal sign flips inside the support',
        'f-space L2 error (eps_rel) is peak-dominated and blind to tail/log accuracy — eps_log (RMS of log-space residual) is the honest metric',
        'Conclusion: need a learned, structural correction on top of the Hermite compression, not just a bigger truncation order',
    ], fontsize=12.5)
    add_image(fig, os.path.join(PLOTS, 'parseval_672.png'), [0.50, 0.08, 0.46, 0.76],
             caption='Parseval convergence: eps_rel keeps improving while eps_log does not')
    pdf.savefig(fig)
    plt.close(fig)


def architecture_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '4. ML Corrector Design')
    bullets = [
        'Variant B: an MLP predicts a per-voxel LOG-space delta correction',
        '  f_final(v) = f_rec_full(v) · exp(Δ_pred(v)) inside sparse_mask, else 0 (structural)',
        'Multiplicative form guarantees positivity regardless of network output — no clipping, no post-hoc fix-up needed',
        'Still "compression": the full tetrahedral decomposition (order 22) IS the compression step; the corrector is a small SHARED model that removes artifacts on top of it, conditioned on cheap already-stored low-order coefficients',
        'Per-cell adaptive regularization on Δ_pred:',
        '  λ_cell = λ_max / (1 + (mean_sq_cell / ref_var)^power)',
        '  suppresses correction on already-good cells without limiting it on genuinely hard ones',
    ]
    add_bullets(fig, [0.06, 0.08, 0.9, 0.78], bullets, fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def feature_evolution_intro_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '5. Feature Evolution',
                 'What information does the MLP need to tell "this looks wrong" from '
                 '"this is genuinely near-empty"?')
    bullets = [
        'Step 1 — low-S coefficients (C_low, S_low≤2, 10 numbers): global but very coarse; cheap, already computed as part of the compressed representation',
        'Step 2 — 1-D Hermite-space power spectra (3×(order+1)=69 numbers): cheap (pure reduction over already-computed coefficients); strong global "how non-Maxwellian is this cell" signal',
        'Step 3 — local 3×3×3 patch of log(f_rec_full) around the query voxel (27 numbers): gives spatial context to localize and shape the correction, but alone cannot disambiguate a genuinely empty tail from a wrongly-zeroed real density (both look like a flat patch at the sparsity floor)',
        'Best result: patch + 1-D spectra COMBINED — each covers the other\'s blind spot',
    ]
    add_bullets(fig, [0.06, 0.08, 0.9, 0.78], bullets, fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def feature_lowS_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '5a. Feature 1 — Low-Order Coefficients',
                 'C_low alone is a coarse, global summary — not enough to arbitrate '
                 'ambiguous per-voxel patch patterns')
    add_image(fig, os.path.join(PLOTS, 'feature_lowS.png'), [0.10, 0.06, 0.80, 0.78])
    pdf.savefig(fig)
    plt.close(fig)


def feature_spectra_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '5b. Feature 2 — 1-D Hermite-Space Power Spectra',
                 'sum_(l,m) C[l,m,n]^2 per axis: cheap and gives a strong, clean signal '
                 'separating Maxwellian-like from strongly non-Maxwellian cells')
    add_image(fig, os.path.join(PLOTS, 'feature_axis_spectra.png'), [0.03, 0.05, 0.94, 0.80])
    pdf.savefig(fig)
    plt.close(fig)


def feature_patch_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '5c. Feature 3 — Local 3×3×3 Patch',
                 'Concatenated into the MLP input (not a real convolution) — gives spatial '
                 'context to localize Gibbs sign-flip artifacts')
    add_image(fig, os.path.join(PLOTS, 'feature_patch.png'), [0.06, 0.06, 0.88, 0.78])
    pdf.savefig(fig)
    plt.close(fig)


def results_within_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '6. Within-Snapshot Results',
                 'Held-out cells of the training bulk file (bulk.0000024) — raw reconstruction '
                 '(left) vs. corrected (right), 2-D projections')
    add_image(fig, os.path.join(PLOTS, 'correction_2d_704.png'),
             [0.02, 0.06, 0.47, 0.78], caption='cid=704 — hard, non-Maxwellian cell')
    add_image(fig, os.path.join(PLOTS, 'correction_2d_1008.png'),
             [0.50, 0.06, 0.47, 0.78], caption='cid=1008')
    pdf.savefig(fig)
    plt.close(fig)


def results_within_slide2(pdf):
    fig = new_page()
    add_title_bar(fig, '6. Within-Snapshot Results (cont.)',
                 '1-D cuts of the integrated VDF — the correction visibly removes ringing '
                 'while tracking the true profile')
    add_image(fig, os.path.join(PLOTS, 'correction_1d_704.png'),
             [0.02, 0.06, 0.47, 0.78], caption='cid=704')
    add_image(fig, os.path.join(PLOTS, 'correction_1d_2016.png'),
             [0.50, 0.06, 0.47, 0.78], caption='cid=2016 — near-Maxwellian lobe cell (correction ≈ 0, as it should)')
    add_bullets(fig, [0.06, 0.0, 0.88, 0.09], [
        'Mean improvement on held-out cells (patch + 1-D spectra, current checkpoint): 50.8% reduction in eps_log',
    ], fontsize=12.5)
    pdf.savefig(fig)
    plt.close(fig)


def cross_timestep_problem_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '7. Cross-Timestep Generalization: the Failure',
                 'A corrector trained on ONE snapshot (bulk.0000024) evaluated on an unseen '
                 'timestep (bulk.0000111)')
    bullets = [
        'Within-snapshot (held-out cells, same file): 62.2% mean improvement, near-perfect lobe-cell handling (eps_log_corr = 0.0107)',
        'Cross-timestep (entirely different, unseen snapshot): mean improvement = −28.0% — the corrector makes things WORSE on average',
        'Diagnosis: the patch-driven correction overfit to the specific non-Maxwellian signature of the single training snapshot; the axis-spectra "triviality recognition" generalized fine on its own',
        'Root cause: single-snapshot training data cannot separate a genuine physical pattern from an artifact of that one moment in time',
    ]
    add_bullets(fig, [0.06, 0.10, 0.9, 0.70], bullets, fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def cross_timestep_fix_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '7. Cross-Timestep Generalization: the Fix',
                 'Retrained on 3 spread-out timesteps (bulk.0000024/50/80), validated on a 4th, '
                 'never-seen timestep (bulk.0000111)')
    add_bullets(fig, [0.06, 0.55, 0.42, 0.30], [
        'Before (1 training snapshot): −28.0% mean improvement',
        'After (3 training snapshots):  +42.8% mean improvement',
        'Best-checkpoint selection now uses the held-out timestep’s',
        '  loss directly (true cross-timestep early stopping)',
    ], fontsize=14)
    add_image(fig, os.path.join(PLOTS, 'validate_2d_464.png'), [0.50, 0.05, 0.47, 0.76],
             caption='cid=464, unseen timestep — median-difficulty cell, corrected')
    pdf.savefig(fig)
    plt.close(fig)


def cross_timestep_fix_slide2(pdf):
    fig = new_page()
    add_title_bar(fig, '7. Cross-Timestep Fix — More Examples',
                 'Unseen-timestep validation cells spanning the difficulty range')
    add_image(fig, os.path.join(PLOTS, 'validate_2d_16.png'),
             [0.02, 0.06, 0.47, 0.78], caption='cid=16 — lobe-like, easiest')
    add_image(fig, os.path.join(PLOTS, 'validate_2d_1152.png'),
             [0.50, 0.06, 0.47, 0.78], caption='cid=1152 — hardest cell in the validation set')
    pdf.savefig(fig)
    plt.close(fig)


def conclusions_slide(pdf):
    fig = new_page()
    add_title_bar(fig, '8. Conclusions and Open Weaknesses')
    bullets = [
        'A small shared MLP, conditioned on cheap already-computed Hermite features, removes most Gibbs ringing without breaking positivity',
        'Combining local spatial context (patch) with global shape context (1-D spectra) was necessary — neither alone solved both failure modes',
        'Multi-snapshot training is REQUIRED for real generalization: a single training snapshot looks great in-sample (62%) but fails out-of-sample (−28%)',
        'Weakest point in the architecture: the regularization strength (λ_max, λ_power, ref_var) is still tuned empirically, per dataset',
        'Next steps: broader multi-snapshot / multi-simulation training sets, automated regularization selection, from-scratch (non-PyTorch) inference reimplementation for in-situ use inside Vlasiator',
    ]
    add_bullets(fig, [0.06, 0.10, 0.9, 0.72], bullets, fontsize=15)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(OUT_PDF) as pdf:
        title_slide(pdf)
        agenda_slide(pdf)
        reconnection_slide(pdf)
        vdf_examples_slide(pdf)
        vdf_cuts_slide(pdf)
        problem_slide(pdf)
        problem_slide2(pdf)
        architecture_slide(pdf)
        feature_evolution_intro_slide(pdf)
        feature_lowS_slide(pdf)
        feature_spectra_slide(pdf)
        feature_patch_slide(pdf)
        results_within_slide(pdf)
        results_within_slide2(pdf)
        cross_timestep_problem_slide(pdf)
        cross_timestep_fix_slide(pdf)
        cross_timestep_fix_slide2(pdf)
        conclusions_slide(pdf)
    print(f"Saved presentation -> {OUT_PDF}")


if __name__ == '__main__':
    main()
