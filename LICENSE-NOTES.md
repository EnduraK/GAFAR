# Licence & IP notes

CRANIUM's own code in this repo is yours (add your chosen licence). The notes
below are about **dependencies and models**, because model licensing is where
biometric prototypes quietly fail due diligence.

## Safe for commercial use
- **dlib** — Boost Software License. Code *and* the ResNet face-recognition
  model are usable in closed-source commercial products. This is the reason
  `DlibEmbedder` is the recommended real backend for the Python engine.
- **MediaPipe** — Apache-2.0. The `face_landmarker.task` model is Apache-2.0.
- **OpenCV (opencv-python)** — Apache-2.0.
- **NumPy** — BSD.
- **face-api.js (@vladmandic/face-api)** — MIT (used only in the browser demo).

## NOT clean for commercial use without action
- **InsightFace / ArcFace** — the *code* is MIT, but the pretrained **model
  weights are non-commercial / research-only**. Do not ship those weights in a
  product. If you want ArcFace-grade accuracy commercially, train your own
  weights or obtain a commercial licence. (This is called out in the concept
  brief and is a real acquisition-diligence trap.)
- **CelebA-Spoof, RMFRD, and several PAD datasets** — research-only licences.
  Fine for experiments; production models must be trained on data you have the
  rights to. Your own collected masked/PPE dataset is both the clean-rights
  path and a defensible asset.

## Model files
- `models/face_landmarker.task` was fetched from the Hugging Face mirror
  `trysem/facelandmarker` (identical 3,758,596-byte MediaPipe float16 model).
  Source of truth is Google's MediaPipe model card (Apache-2.0).

## Reminder
`StubEmbedder` and the browser demo's face-api descriptor are **stand-ins** to
prove the pipeline. Neither is the periocular-specific model the product needs —
that model is Phase 1 of the build plan.
