import sys, torch, numpy as np, os
from rl import train
fuzzy = None if sys.argv[1] == "none" else sys.argv[1]
rkind, seed, eps = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tag = sys.argv[5] if len(sys.argv) > 5 else "main"
os.makedirs(f"models/{tag}", exist_ok=True)
if os.path.exists(f"models/{tag}/{sys.argv[1]}_{rkind}_s{seed}.pt"):
    print("skip, already done"); sys.exit(0)
actor, hist = train(fuzzy, rkind, seed, eps, log=lambda m: print(m, flush=True))
name = f"models/{tag}/{sys.argv[1]}_{rkind}_s{seed}"
torch.save(actor, name + ".pt"); np.save(name + "_hist.npy", hist)
print("saved", name, flush=True)
