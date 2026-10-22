"""HE1 pre-flight (CPU, login node): adapter hashes against their records, which PB2 adapters survive, disk.

  he1_preflight.py      -> he1/results/preflight.json (printed summary)
Stop rules (handoff §8): a hash mismatch against a record that exists; a missing adapter with no listed substitute.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import he1_common as E  # noqa: E402


def main():
    out = {"adapters": {}, "pb2": {}, "disk_gb": {}}
    stop = []
    for m, (adir, _, src, words) in E.MODELS.items():
        f = adir / "adapter_model.safetensors"
        want = E.expected_sha(m)
        if not f.exists():
            out["adapters"][m] = {"path": str(adir), "exists": False, "record": src}
            if m != "Rpet-F":
                stop.append(f"{m} missing")
            continue
        got = E.sha256(f)
        ok = None if want is None else got == want
        cfg = json.loads((adir / "adapter_config.json").read_text())
        out["adapters"][m] = {"path": str(adir), "exists": True, "sha256": got, "expected": want, "record": src, "match": ok,
                              "r": cfg["r"], "alpha": cfg["lora_alpha"], "rslora": cfg.get("use_rslora"), "words": words,
                              "bytes": f.stat().st_size}
        if ok is False:
            stop.append(f"{m} sha256 mismatch")
        print(f"{m:10s} {got[:12]}… expected {want[:12] + '…' if want else 'none (recorded now)':24s} match={ok} r={cfg['r']} "
              f"alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora')}", flush=True)
    out["pb2"] = {"folder": str(E.PB2C), "exists": E.PB2C.exists(),
                  "R-every": (E.PB2C / "R-every" / "final" / "adapter_model.safetensors").exists(),
                  "R-adv1": (E.PB2C / "R-adv1" / "final" / "adapter_model.safetensors").exists(),
                  "R0": (E.PB2C / "R0" / "final" / "adapter_model.safetensors").exists()}
    tw_adv1 = "Rpet-F" if out["adapters"]["Rpet-F"].get("exists") else "Rev-F (other anchor) unless AN1's Rpet-F exists when Part B starts"
    out["twins"] = {"twin(K-every)": "R-every (PB2)" if out["pb2"]["R-every"] else "Rev-F (PB3)",
                    "twin(K-adv1)": "R-adv1 (PB2)" if out["pb2"]["R-adv1"] else tw_adv1,
                    "R0": "R0 (PB2)" if out["pb2"]["R0"] else "R0-F (PB3)"}
    for p in ("/work/u4161854", "/home/u4161854"):
        out["disk_gb"][p] = E.disk_free_gb(p)
    out["stop"] = stop
    E.write_json(E.RESULTS / "preflight.json", out)
    print(json.dumps({"pb2": out["pb2"], "twins": out["twins"], "disk_gb": out["disk_gb"], "stop": stop}, indent=1))


if __name__ == "__main__":
    main()
