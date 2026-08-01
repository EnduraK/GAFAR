# CRANIUM — MVP

Upper-face (periocular) verification and an authorisation engine that keeps
working when the lower face is covered — surgical masks, respirators, helmets,
veils. This is the Phase 1 prototype from the concept brief.

## Why CRANIUM — the origin

CRANIUM carries my father's name and his idea. **GAFAR** is his first name.
Since he was a child he was told that a future technological advance would
centre on the **forehead** — he was never sure whether it meant the skin, the
skull, or the whole structure of that region. **CRANIUM** is that region: the
house of components he was pointing at. This project takes the idea literally —
identity verification that works from the upper face, forehead and eyes — and
carries his name.

## What this is

A solo student portfolio and learning project (I'm at CU London, Dagenham),
built to push the idea as far as it will go. The goal is general **role-based
access control and authentication** — confirming the right person before a
sensitive action — with **finance** as the lead example and any high-stakes or
covered-face context fair game. It doubles as a way to learn new libraries and
techniques, and it could grow into something real. We'll see how it goes.

**The thesis, in one picture** (`docs/periocular_proof.png`): the yellow box is
the region CRANIUM matches on — eyes, brows, nose bridge, forehead. Everything
below it (nose, mouth, jaw) can be covered without touching the matched region.

CRANIUM is **1:1 verification only** — a probe is compared against the single
enrolled template for the claimed identity, never searched against a database.
That is a deliberate design choice: it keeps the system inside the EU AI Act's
biometric-*verification* carve-out (Art 3(36) / Annex III), out of the
high-risk remote-identification regime.

---

## What's in the box

```
cranium/            the engine (importable package)
  landmarks.py        MediaPipe FaceLandmarker wrapper -> 478 landmarks
  periocular.py       upper-face ROI + crop from landmarks
  embedding.py        pluggable embedders: StubEmbedder (wiring), DlibEmbedder (real)
  verifier.py         1:1 enroll/verify, cosine match, template store (vectors only)
  liveness.py         blink / attention challenge (EAR state machine) — v0
  authorization.py    the authorisation-event engine + hash-chained audit log
  pipeline.py         image -> landmarks -> crop -> embedding
  config.py           thresholds + per-risk policies (all placeholders)
  cli.py              enrol / verify / authorise  (webcam or --image)
demo/cranium_demo.html   single-file browser demo (webcam, live)
models/face_landmarker.task   bundled landmark model (~3.7 MB)
tests/              31 tests (pytest) incl. a real-face integration test
docs/periocular_proof.png     the proof figure
```

## Quick start — browser demo (no install)

Open `demo/cranium_demo.html` in Chrome/Edge/Safari and click **Enable camera**.
It must be served over `https://` or `localhost` (browsers only give camera
access in a secure context). The simplest way:

```bash
cd demo && python3 -m http.server 8000
# then open http://localhost:8000/cranium_demo.html
```

Enrol your face, pick **Medication cabinet** or **High-value transfer**, choose
a risk level, and hit **Request authorisation**. Standard+ risk runs a live
**blink** liveness check. Every decision is written to a tamper-evident,
hash-chained audit log. Models load once from a public CDN.

> The browser descriptor is a general **full-face** model (face-api.js) used as
> a stand-in. It demonstrates real 1:1 matching, the periocular ROI, liveness,
> and the authorisation flow — it is not the periocular model the product needs.

## Quick start — Python engine

```bash
pip install -r requirements.txt          # numpy, opencv, mediapipe
# enrol from the webcam, then authorise a controlled-drug release:
python -m cranium.cli enrol     --subject alice --webcam
python -m cranium.cli authorise --subject alice --webcam \
       --action release_controlled_drug --resource "Morphine 10mg" --risk standard
```

Headless (images instead of a camera) works too, with `--image a.png b.png …`.
For **real recognition** (not the stub), use the dlib backend:

```bash
python -m cranium.cli --backend dlib \
   --dlib-shape shape_predictor_5_face_landmarks.dat \
   --dlib-rec   dlib_face_recognition_resnet_model_v1.dat \
   enrol --subject alice --webcam
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest            # 31 passed
```

The suite unit-tests the logic that can actually be wrong — verifier maths and
thresholds, the authorisation state machine across risk levels, the audit hash
chain (including tamper detection), ROI geometry, and the blink challenge — and
runs the **real** MediaPipe pipeline on a bundled photograph to prove
landmarks → periocular crop → embedding end to end.

## What is and isn't validated

**Validated here:** all engine logic (31 tests green); the real CV pipeline on a
real face; the CLI enrol/verify/authorise flows; the browser demo boots and its
UI logic runs headless.

**Needs real captures / your webcam:** embedding-model *accuracy* (EER), live
liveness robustness, and the browser demo's camera path. The stub embedder is
deterministic, not a recogniser — accuracy numbers are meaningless until a real
model runs on collected data. That measurement (target **EER < 3%** on
webcam-quality captures) is the Phase 1 gate.

## Honest limitations (see the concept brief for the full risk register)

- Periocular matching carries roughly a 10× error penalty vs full-face; it
  needs constrained capture and an always-present second factor (badge/username).
- Sunglasses defeat it; goggles/anti-fog degrade it.
- The blink check is **not** presentation-attack detection. Real liveness is a
  layered Phase 2 job (passive CNN + attention + active screen-illumination
  reflection) with injection-attack defence, and it lives in a patent minefield
  (freedom-to-operate search required before shipping active illumination).

## Licence & models
See `LICENSE-NOTES.md`. Short version: dlib and MediaPipe models are
commercially clean; InsightFace/ArcFace weights are **not** — don't ship them.
