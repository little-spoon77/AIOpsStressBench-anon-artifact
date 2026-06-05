# Foundation Reference Failures / Scope

- Chronos-Bolt tiny/small/base completed on A6000 GPU1 with local-uploaded HuggingFace cache.
- The execution machine had no outbound network access; model caches were staged before evaluation.
- TimesFM and Moirai were not run in this A6000 pass. Local installation attempts previously failed under Python 3.13 dependency constraints, and the A6000 execution machine had no outbound network for direct model/dependency acquisition. They remain optional reference-only extensions, not part of the core benchmark claim.
- No fine-tuning was performed; all rows are bounded zero-shot reference results.

