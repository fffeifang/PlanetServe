# Fraction of malicious nodes

### Anon. vs. malicious frac
```bash
python anon.py \
  --N 10000 \
  --f-values 0.001,0.01,0.05,0.1,0.2,0.3,0.5 \
  --output-dir out
```

### Conf. vs. malicious frac
```bash
python conf.py \
  --N 10000 --runs 50000 --seed 12345 \
  --f-min 0.001 --f-max 0.1 --f-points 10 \
  --gc-n 4 --gc-k 3 --gc-L 6 \
  --ps-n 4 --ps-k 3 --ps-L 4 \
  --output-dir out
```