"""Export trained actors to MATLAB .mat files (weights for ctrl_ddpg.m).
usage: python3 export_policies.py <tag> [model names ...]  (default: all .pt in models/<tag>)"""
import sys, glob, os, torch, numpy as np
from scipy.io import savemat
from rl import Actor, Critic, Fuzzify  # noqa
tag = sys.argv[1]; names = sys.argv[2:]
files = [f"models/{tag}/{n}.pt" for n in names] if names else sorted(glob.glob(f"models/{tag}/*.pt"))
os.makedirs("matlab/policies", exist_ok=True)
for f in files:
    a = torch.load(f, weights_only=False); sd = {k: v.detach().numpy().astype(float) for k, v in a.state_dict().items()}
    d = {"W0": sd["net.0.weight"], "b0": sd["net.0.bias"], "W1": sd["net.2.weight"], "b1": sd["net.2.bias"],
         "W2": sd["net.4.weight"], "b2": sd["net.4.bias"]}
    if "f.c" in sd: d["c"] = sd["f.c"]; d["log_s"] = sd["f.log_s"]
    out = "matlab/policies/" + os.path.basename(f)[:-3] + ".mat"; savemat(out, d); print("exported", out)
