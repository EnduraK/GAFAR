# Measuring the periocular EER — the Phase 1 gate

The Phase 1 question is simple: **does the engine actually recognise people from the
upper face?** The single number that answers it is the **Equal Error Rate (EER)** —
the balanced point where the false-accept rate equals the false-reject rate. Lower
is better. The gate in the concept brief is **EER < 3%** on webcam-quality captures.

## What the harness does

`cranium/evaluation.py` is backend-agnostic. Give it identity labels and embeddings
and it returns EER, the operating threshold, AUC, and TAR@FAR (true-accept rate at
fixed false-accept rates). `scripts/eval_periocular.py` runs the whole pipeline —
image → MediaPipe landmarks → crop → embedding → EER — for **full face vs periocular
crop**, so you see the upper-face-only penalty directly.

## Run it

```bash
pip install facenet-pytorch scikit-image scikit-learn   # facenet = real face model, MIT

# public baseline (no consent needed): Labelled Faces in the Wild
python scripts/eval_periocular.py --backend facenet --max-images 400

# your real target: your 20-volunteer set, one folder per person
#   volunteers/alice/01.jpg volunteers/alice/02_mask.jpg volunteers/bob/01.jpg ...
python scripts/eval_periocular.py --dataset folder --images volunteers/
```

It prints both EERs, the periocular/full-face penalty ratio, a PASS/NOT-YET against
the 3% gate, and saves a genuine-vs-impostor score-distribution figure.

## The pitfall to avoid (important)

Do **not** reach for a stock **MobileNet/EfficientNet** off the shelf. Those are
ImageNet *classifiers* — they were never trained to tell people apart, so their
features don't discriminate identity and the EER will be poor. Use a model trained
on faces with a metric-learning objective. The clean, commercially-usable choice is
**facenet-pytorch** (MIT, InceptionResnetV1 on VGGFace2), which is what
`FacenetEmbedder` wraps. `dlib` (Boost licence) is also a real face embedder and a
fine second baseline.

## How to read the result

- **Full-face EER** is your ceiling — how well the chosen model does with the whole
  face. Expect low single digits on a clean set.
- **Periocular EER** is the product-relevant number. It will be higher, because
  you've hidden the nose and mouth. The *ratio* is the honest cost of the mask-proof
  property.
- If periocular EER is already under ~3%, the stand-in model is good enough to keep
  building the demo around. If it's higher (likely at first), that's the signal to
  **fine-tune a periocular model** (the real Phase 1 engineering) on UFPR-Periocular
  and RMFRD, or to fuse periocular with whatever full face is visible.

## Honesty about the estimate

EER from ~20 volunteers is a **rough signal, not a certified number**. A trustworthy
figure needs many identities and many pairs (the reason NIST/iBeta use large,
controlled sets). Report your sample size alongside the EER every time, and treat
sub-3% on a tiny set as "promising," not "proven." Certification (Phase 3) is where
the audited number comes from.

## Good practice when you collect data

- Multiple captures per person, and across 2–3 different webcams — cross-device
  variation is where naive models fall over.
- Both masked and unmasked, plus glasses if relevant, so you measure the conditions
  you actually claim to handle.
- Consistent framing/distance, decent lighting — but not *too* clean, or you'll
  overstate real-world accuracy.
- Measure demographic balance (skin tone, age, sex) and report per-group EER; a
  single average can hide bias, and bias testing is a procurement requirement.
- Collect it under proper consent — see `BIOMETRIC_CONSENT_FORM.md`.
