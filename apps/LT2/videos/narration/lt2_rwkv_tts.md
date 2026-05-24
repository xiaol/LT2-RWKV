# LT2-RWKV Short Explainer Narration

Host A voice: `English_captivating_female1`
Host B voice: `English_CaptivatingStoryteller`
MiniMax model: `speech-2.8-hd`
Speed: `1.0`
Pitch: `0`
Volume: `1.0`

## 0:00-0:14
A: (breath) What if a looped Transformer used RWKV-7 as its recurrent mixer, instead of GDN?
B: (chuckle) That sounds like a small code change with a big architectural question.

## 0:14-0:38
A: First, the relation. A Universal Transformer reuses the same transition block across depth steps.
B: So a looped Transformer is the language-model version of that same recurrence idea?
A: Exactly. It repeatedly applies a shared Transformer block. LT2 asks what efficient token mixer should live inside that loop.

## 0:38-0:55
A: Each loop pass adds effective computation without adding a new copy of the parameters.
B: The shared block sees the updated hidden state again and again.
A: Right. The attraction is more recurrent refinement at nearly the same model size.

## 0:55-1:23
A: The LT2 baseline uses GDN, a DPLR-style recurrent mixer. I added a full RWKV-7 native block, with time mix and channel mix, as its replacement.
B: (breath) Same looping idea, but a more expressive recurrent core.
A: In the repo, the switch is `layer_pattern = "rwkv7_native"`.

## 1:23-1:50
A: Then I ran a parameter-matched real-token test on FineWeb sp ten twenty-four.
B: Nearly the same parameters?
A: Yes. GDN has nine hundred sixty-two thousand parameters; RWKV-7 has nine hundred fifty-nine thousand.
B: And RWKV-7 reached about one point five seven times the throughput, with lower final validation loss.

## 1:50-2:12
A: Look at the smoothed training curve. In this five-thousand-step ablation, RWKV-7 drops faster and stays below GDN.
B: (chuckle) It is still an early small-scale run, not the final pretraining verdict.
A: Correct. But it is a strong reason to run the full comparison.

## 2:12-2:33.92
A: <#0.3#> The hypothesis is simple: looping already benefits recurrent mixers, and RWKV-7 may be a better recurrent core for LT2.
B: Give RWKV-7 a try. The code and curve are at github dot com slash xiaol slash LT2 dash RWKV.
