# CRANIUM demo — HCI & usability design notes

This documents how the demo interface (`demo/cranium_demo.html`) is designed
against **Nielsen's 10 usability heuristics** and a few broader HCI principles.
Each heuristic below names the concrete feature that implements it, so the design
choices are traceable (and defensible in a review or viva). Source comments in
the HTML are tagged `[H1]…[H10]` at the relevant code.

## Nielsen's 10 heuristics → where each lives in the UI

| # | Heuristic | How CRANIUM applies it |
|---|-----------|------------------------|
| 1 | **Visibility of system status** | A single status line (`role="status" aria-live="polite"`) always says what's happening — "Looking for your face", "Face detected", "Enrolled ✓", "Live check — please blink", "Access granted/denied". A model-loading progress bar, a 3-step tracker, and a live match-confidence meter all reflect current state continuously. |
| 2 | **Match between system and the real world** | Plain language, not jargon: "secure access by your face", "Security level", "Live match to your enrolled face", "Please blink once". The score is shown as **Strong / Good / Weak / No match (92%)**, not "cosine 0.86". Real-world scenarios (medicine cabinet, money transfer) frame the abstract "authorisation event". |
| 3 | **User control & freedom** | Clear exits everywhere: **Clear** enrolment, **Cancel** during the live check, **Camera on** toggle, **Clear log**. No action traps the user; the live-check can be abandoned and returns to a clean state. |
| 4 | **Consistency & standards** | One button system (primary / secondary / small / danger), consistent colour semantics (teal = go, green = success, amber = attention, red = denied), sentence case throughout, standard form controls, familiar tab pattern for scenarios. |
| 5 | **Error prevention** | Invalid actions are disabled with a reason rather than allowed-then-rejected: **Request access** is disabled until a face is enrolled and visible, with the hint "Enrol your face first (step 2)". Destructive actions (**Clear**, **Clear log**) require a confirming second click ("Click again to confirm"). |
| 6 | **Recognition rather than recall** | Nothing to memorise: the 3-step tracker shows where you are, the current security policy is spelled out in a sentence ("Standard access needs a good match and a live blink check"), and ⓘ tooltips explain each term in place. The current threshold is drawn as a marker on the confidence meter. |
| 7 | **Flexibility & efficiency of use** | Novices pick a **Security level** and the threshold is set for them (accelerator); experts open **Advanced** to see and tune the raw cosine threshold. Keyboard shortcuts: `E` to enrol, `Enter` to request access. |
| 8 | **Aesthetic & minimalist design** | Progressive disclosure keeps the default view calm — technical detail (cosine, threshold slider, raw score) is collapsed inside **Advanced**; the "How it works" panel is collapsed by default. One primary action is emphasised at a time as the flow advances. |
| 9 | **Help users recognise, diagnose & recover from errors** | Errors are plain and actionable, never codes: a blocked camera shows "We can't see your camera. Allow camera access… then Retry" with a **Retry** button; a failed match explains "Face didn't match closely enough (71%, need 86%)"; a denial offers **Try again** and **Check my face**. |
| 10 | **Help & documentation** | A one-click **How it works** panel explains the three steps, the privacy model, and the honest "stand-in model" caveat. Inline ⓘ tooltips document Security level, Live match, and the audit log without leaving the page. |

## Broader HCI principles

**Accessibility.** Visible keyboard focus rings; status changes announced to
screen readers via `aria-live` regions; tooltips reachable by keyboard
(`tabindex="0"`, `role="img"` with the text in `aria-label`/`data-tip`); semantic
landmarks and labelled controls; `prefers-reduced-motion` disables animation;
colour is never the *only* signal (icons + text accompany every state).

**Feedback & response time (Norman's gulf of evaluation).** Every user action
produces immediate visible feedback — enrol, threshold drag, scenario switch, and
the per-frame confidence meter all update live, so the user can always tell what
the system understood.

**Progressive disclosure & Hick's Law.** The first screen offers few choices;
complexity (thresholds, raw scores) is revealed only on request, reducing
decision time and perceived difficulty.

**Fitts's Law.** The primary action is a full-width button; related controls are
grouped so pointer travel is short.

**Gestalt grouping.** Cards, dashed request boxes, and consistent spacing group
related controls ("Your identity" vs "Access request" vs "Access log") so
structure is read at a glance.

**Honesty / trust (ethical HCI).** The interface never overstates the prototype:
it states plainly that a stand-in model is used and that the mask-proof version is
still to come, and it foregrounds the privacy model ("nothing leaves your
device"). Trust is earned by not faking capability.

## Known gaps / recommended next evaluation

This is heuristic-informed *design*, not yet heuristic *evaluation*. Before real
users, run:
1. A formal **heuristic evaluation** with 3–5 evaluators, scoring each issue on
   Nielsen's 0–4 severity scale.
2. A small **think-aloud usability test** (5 users) on the core task "enrol, then
   release a medicine" — measure task success, time, and error count.
3. An **accessibility audit** (keyboard-only pass, screen-reader pass, automated
   axe/Lighthouse run, colour-contrast check to WCAG AA).
4. Localisation review of all plain-language strings.
