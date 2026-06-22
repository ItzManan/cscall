# Phase 1 fine-tune eval results (2026-06-15)

## Full GPU compare — 664 HiACC test clips (faster-whisper float16, language=None)

| Group | Baseline WER | Fine-tuned WER | Δ WER |
|---|---|---|---|
| **overall** | 0.485 | 0.502 | −0.017 |
| high | 0.593 | 0.591 | +0.002 |
| low  | 0.634 | 0.746 | −0.113 |
| mid  | 0.814 | 0.818 | −0.004 |
| none | 0.155 | 0.186 | −0.030 |

Verdict: no improvement. Negative Δ = fine-tuned worse, but effect is within
GPU/CPU + temperature-fallback noise (see findings doc).

## 60-clip regime probe (CPU int8; none=34, high=19, mid=4, low=3)

```
                overall   high(19)   mid(4)    none(34)
base(auto)      0.407     0.486      0.863     0.238
base(hi)        0.843     0.482      0.863     1.206
ft(auto)        0.380     0.498      0.863     0.146
ft(hi)          0.695     0.490      0.863     0.872
```

Verdict: forcing language=hi is a net loss (English-heavy set); fine-tune
dormant under auto-detect; no movement on high/mid code-switch buckets.
