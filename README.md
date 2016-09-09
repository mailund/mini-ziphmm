# mini-ziphmm

This is a minimal reimplementation of zipHMM in python.
Also contains cython and weave implementations of standard hidden markov models
for comparison.

All very rough and not optimized. The preprocessing step in particular is slow.

To run:

```bash
make
python test.py
```
