// Formal project report (max. 5 pages). Build: python scripts/build_report_pdf.py

#let title = "Underwater Image Enhancement Using Classical Image Processing Techniques"
#let author = "Emre Yoleri"
#let date = "October 2026"

#set document(title: title, author: author)
#set page(paper: "a4", margin: (x: 2.5cm, top: 2.5cm, bottom: 2.3cm),
  footer: context align(center, text(size: 9pt, counter(page).display("1"))))
#set text(font: "Times New Roman", size: 11pt, lang: "en", region: "us")
#set par(justify: true, leading: 0.62em, spacing: 0.95em, first-line-indent: 0pt)
#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading: set block(above: 1.2em, below: 0.7em)
#set math.equation(numbering: "(1)")
#set figure(gap: 0.6em)
#show figure: set block(above: 1em, below: 1em)
#show figure.caption: set text(size: 9.5pt)
#show figure.caption: it => [*#it.supplement #context it.counter.display(it.numbering).* #it.body]
#show figure.where(kind: table): set figure.caption(position: top)
#set table(stroke: (x, y) => (top: if y <= 1 { 0.6pt } else { 0pt }, bottom: 0.6pt), inset: (x: 5pt, y: 3.2pt))
#show table: set text(size: 9.5pt)
#show table.cell.where(y: 0): strong

// ---------------------------------------------------------------- title block
#align(center)[
  #text(size: 15pt, weight: "bold")[#title]
  #v(0.5em)
  #text(size: 11pt)[#author]
  #v(0.1em)
  #text(size: 10pt, style: "italic")[Technical Report · #date]
]
#v(0.6em)
#line(length: 100%, stroke: 0.5pt)
#block(inset: (x: 0.6cm))[
  #set text(size: 10pt)
  *Abstract.* Underwater images suffer from a blue–green colour cast and low contrast because water
  absorbs red light and scatters light along the line of sight. This report presents an image
  processing pipeline in Python and OpenCV that corrects both problems without any training data. Five
  classical techniques — histogram equalization (on RGB and on luminance), CLAHE, Gray-World white
  balance and red-compensated white balance — are compared with a combined pipeline on 100 image pairs
  from the public UIEB dataset, using PSNR, SSIM, UCIQE and UIQM. The combined pipeline achieved the
  best PSNR (19.72 dB, +2.03 dB over the input), SSIM (0.834) and UIQM (2.954), with statistically
  significant gains over the input on all four metrics (Wilcoxon, p < 0.003), at about 12 ms per image
  on a CPU.

  *Keywords:* underwater image enhancement, CLAHE, white balance, histogram equalization, image quality.
]
#line(length: 100%, stroke: 0.5pt)

// ---------------------------------------------------------------- body
= Introduction

Underwater images are used in marine biology, inspection of submerged structures and underwater
robotics. Their quality is limited by two physical effects. First, water absorbs long wavelengths much
faster than short ones, so red light disappears within a few metres and images take on a blue–green
cast @akkaynak2018. Second, suspended particles scatter light, which adds a haze veil that lowers
contrast @jaffe1990. Classical enhancement techniques are fast and need no training, but each one fixes
only part of the problem: contrast enhancement keeps the colour cast, and white balance keeps the low
contrast.

The objective of this work is to (i) implement and compare standard techniques — histogram
equalization, CLAHE and white balance — on a public underwater dataset, (ii) combine them into a single
pipeline that corrects colour and contrast together, and (iii) evaluate the results visually and with
image quality metrics.

= Methodology

== Dataset

The Underwater Image Enhancement Benchmark (UIEB) @li2020uieb contains 890 real underwater images.
Each image has a reference version that volunteers chose as the best of several enhancement results,
which allows full-reference evaluation. A random subset of 100 image pairs (fixed seed) was used. All
images were resized so that the longest side is at most 640 pixels.

== Enhancement Methods

Six methods were implemented with OpenCV (@tab-methods). Histogram equalization (HE) maps grey levels
through the cumulative histogram so that the output histogram is approximately flat @gonzalez2018.
CLAHE @zuiderveld1994 applies equalization on small tiles and clips each tile histogram to limit noise
amplification; it was applied to the lightness channel $L^*$ of CIELAB so that colours are not changed.
Gray-World white balance @buchsbaum1980 scales each channel so that all channel means are equal. Because
this over-amplifies an almost empty red channel, the red channel was first compensated from the green
channel, following Ancuti et al. @ancuti2018:

$ I'_r = I_r + alpha (macron(I)_g - macron(I)_r)(1 - I_r) I_g , $ <eq-red>

where intensities are in $[0, 1]$, $macron(I)$ denotes a channel mean and $alpha = 1$.

The *proposed pipeline* corrects colour first and contrast second: (1) red channel compensation
(@eq-red), (2) Gray-World white balance, (3) a per-channel contrast stretch between the 0.5th and 99.5th
percentiles, and (4) CLAHE on $L^*$ with clip limit 2.0 and an 8×8 tile grid.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: left,
    table.header([Method], [Operation], [Corrects]),
    [HE (RGB)], [Global HE on each RGB channel], [colour + contrast],
    [HE (Luminance)], [Global HE on the Y channel (YCrCb)], [contrast],
    [CLAHE], [CLAHE on $L^*$ (clip limit 2.0, 8×8 tiles)], [local contrast],
    [Gray-World WB], [Equalize channel means], [colour],
    [Underwater WB], [Red compensation (@eq-red) + Gray-World], [colour],
    [Proposed pipeline], [Red comp. → Gray-World → stretch → CLAHE], [colour + contrast],
  ),
  caption: [Compared enhancement methods.],
) <tab-methods>

== Evaluation Metrics

Two full-reference metrics compare each result with the UIEB reference: PSNR, based on the mean squared
error, and SSIM @wang2004ssim, which compares local luminance, contrast and structure. Two no-reference
metrics designed for underwater images were also used: UCIQE @yang2015uciqe, a combination of chroma
variation, luminance contrast and saturation, and UIQM @panetta2016uiqm, a combination of colourfulness,
sharpness and contrast measures. Higher values are better for all four. Differences between methods
were tested with the paired two-sided Wilcoxon signed-rank test ($alpha = 0.05$).

= Results

== Quantitative Results

@tab-results shows the mean metrics over the 100 images. The proposed pipeline obtained the best PSNR,
SSIM and UIQM. Its improvement over the raw input was statistically significant for all four metrics
($p < 0.003$). Compared with CLAHE, the strongest single technique, it was significantly better in PSNR
(median +1.25 dB, $p < 10^(-4)$), UCIQE and UIQM, while the SSIM difference was not significant
($p = 0.06$). HE (RGB) achieved the highest UCIQE but a lower PSNR than the unprocessed input. All
methods ran in real time: about 1 ms for HE and CLAHE and 12 ms for the full pipeline (Apple M3 Pro,
single thread).

#figure(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr, 0.8fr),
    align: (left, center, center, center, center, center),
    table.header([Method], [PSNR (dB)], [SSIM], [UCIQE], [UIQM], [Time (ms)]),
    [Raw input], [17.69], [0.774], [0.259], [2.409], [—],
    [HE (RGB)], [16.72], [0.792], [*0.356*], [2.813], [1.1],
    [HE (Luminance)], [16.47], [0.779], [0.338], [2.459], [1.0],
    [CLAHE], [17.95], [0.821], [0.291], [2.744], [1.4],
    [Gray-World WB], [17.36], [0.769], [0.238], [2.466], [4.1],
    [Underwater WB], [17.87], [0.801], [0.228], [2.443], [6.2],
    [*Proposed pipeline*], [*19.72*], [*0.834*], [0.328], [*2.954*], [12.3],
    table.hline(stroke: 0.4pt),
    [_Reference image_], [—], [—], [0.319], [2.780], [—],
  ),
  caption: [Mean results on 100 UIEB images (higher is better; best method in bold).],
) <tab-results>

== Visual Comparison

@fig-visual shows the results on five test images. HE (RGB) removes the cast but produces false red and
cyan colours and amplified noise. HE (Luminance) and CLAHE increase contrast but keep the cast.
Gray-World white balance adds red noise when the red channel is nearly empty (second row), which red
compensation largely prevents. The proposed pipeline gives the most natural colours and the clearest
texture and is visually closest to the references; on the second image it slightly over-corrects the
blue water towards purple.

#figure(
  image("/results/figures_report/visual_comparison.jpg", width: 100%),
  caption: [Before/after comparison on five UIEB images (most degraded at the top). Last column: reference.],
) <fig-visual>

== Ablation and Parameter Sensitivity

Removing each stage of the pipeline in turn (@tab-ablation) shows that red compensation and the
contrast stretch both improve the results. Removing CLAHE, however, gives the highest PSNR and SSIM but
the lowest UIQM. @fig-sens confirms this trade-off: as the CLAHE clip limit grows, PSNR falls steadily
while UIQM rises and levels off above a clip limit of about 2.

#grid(
  columns: (1fr, 1.25fr),
  gutter: 0.6cm,
  align: horizon,
  [#figure(
    table(
      columns: (1.6fr, 0.9fr, 0.8fr, 0.8fr),
      align: (left, center, center, center),
      table.header([Configuration], [PSNR], [SSIM], [UIQM]),
      [Full pipeline], [19.72], [0.834], [*2.954*],
      [w/o red comp.], [19.18], [0.801], [2.925],
      [w/o white bal.], [20.07], [0.837], [2.941],
      [w/o stretch], [18.09], [0.846], [2.880],
      [w/o CLAHE], [*22.12*], [*0.894*], [2.558],
    ),
    caption: [Ablation of pipeline stages.],
  ) <tab-ablation>],
  [#figure(
    image("/results/figures_report/clip_tradeoff.png", width: 100%),
    caption: [Effect of the CLAHE clip limit on PSNR and UIQM.],
  ) <fig-sens>],
)

= Discussion

The results confirm that underwater degradation has two separate causes that need separate
corrections. Each single technique improved only the property it targets, while the pipeline that
corrects colour before contrast was the only method that improved all four metrics at once. Red channel
compensation was essential: without it, Gray-World white balance turned the almost empty red channel
into visible red noise.

Two points need care when interpreting the metrics. First, CLAHE creates a trade-off between fidelity
to the references (PSNR, SSIM) and perceived sharpness and contrast (UIQM). The clip limit therefore
works as a single "detail" control, and lower values should be used when faithful colours matter more
than visible detail. Second, the no-reference metrics reward colourfulness and contrast and do not
penalize over-enhancement: HE (RGB), with obvious false colours, received the highest UCIQE, and several
methods scored above the human-chosen references. No-reference metrics should therefore always be read
together with full-reference metrics and visual inspection.

The main limitations are that the UIEB references are preferred enhancement results rather than true
ground truth, that only 100 of the 890 images were used, and that the global white balance cannot model
the depth-dependent attenuation of light.

= Conclusion

A classical image processing pipeline for underwater images was developed in Python and OpenCV and
compared with five standard techniques on the UIEB dataset. Combining red channel compensation,
Gray-World white balance, contrast stretching and CLAHE gave the best PSNR (19.72 dB), SSIM (0.834) and
UIQM (2.954), with significant improvements over the input, and runs in about 12 ms per image without a
GPU or training data. The method is also available in an interactive Streamlit web application. Future
work includes depth-aware colour correction @akkaynak2019seathru, automatic per-image parameter
selection, and evaluation on the full dataset.

#v(0.3em)
#{
  set text(size: 9.5pt)
  set par(leading: 0.5em, spacing: 0.55em)
  show heading: set block(above: 1.2em, below: 0.7em)
  bibliography("references.yml", title: [References], style: "ieee")
}
