"""MT1 A10 step-alignment logger (handoff §3.2 A10), injected via PYTHONPATH so the YAML stays byte-identical except for
the four keys §6 allows. Inert unless MT1_A10_LOG is set. When it is:

- when transformers.trainer is first imported (also inside accelerate's subprocess), Trainer.__init__ is wrapped to add
  one read-only callback;
- on train begin the callback records len(train_dataset) (A10's 8,192-row assertion) and wraps the train sampler so
  every index it yields is logged, both epochs;
- the log is written to $MT1_A10_LOG at every save and at train end.

It never raises into training: on any failure it writes {"status": "NOT CHECKABLE", "reason": ...} instead.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

_LOG = os.environ.get("MT1_A10_LOG")


def _write(payload):
    try:
        tmp = _LOG + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, _LOG)
    except Exception as e:  # noqa: BLE001
        print(f"[A10] could not write log: {e!r}", file=sys.stderr, flush=True)


def _patch(module):
    try:
        from transformers import TrainerCallback
    except Exception as e:  # noqa: BLE001
        _write({"status": "NOT CHECKABLE", "reason": f"import TrainerCallback: {e!r}"})
        return

    class A10Callback(TrainerCallback):
        def __init__(self):
            self.state = {"status": "logging", "indices": [], "n_train_dataset": None, "sampler": None,
                          "micro_batch_size": None, "gradient_accumulation_steps": None}

        def on_train_begin(self, args, state, control, train_dataloader=None, **kw):
            st = self.state
            try:
                st["micro_batch_size"] = args.per_device_train_batch_size
                st["gradient_accumulation_steps"] = args.gradient_accumulation_steps
                ds = getattr(train_dataloader, "dataset", None)
                st["n_train_dataset"] = len(ds) if ds is not None else None
                sampler = None
                for path in (("batch_sampler", "sampler"), ("sampler",), ("base_dataloader", "batch_sampler", "sampler")):
                    obj = train_dataloader
                    for attr in path:
                        obj = getattr(obj, attr, None)
                    if obj is not None and hasattr(obj, "__iter__") and not isinstance(obj, (list, tuple)):
                        sampler = obj
                        break
                if sampler is None:
                    raise RuntimeError("no sampler found on the train dataloader")
                cls = sampler.__class__
                log = st["indices"]

                def logged_iter(self_):
                    for i in cls.__iter__(self_):
                        log.append(int(i))
                        yield i

                sampler.__class__ = type("A10" + cls.__name__, (cls,), {"__iter__": logged_iter})
                st["sampler"] = cls.__module__ + "." + cls.__name__
            except Exception as e:  # noqa: BLE001
                st["status"] = "NOT CHECKABLE"
                st["reason"] = repr(e)
            _write(st)

        def on_save(self, args, state, control, **kw):
            _write(self.state)

        def on_train_end(self, args, state, control, **kw):
            if self.state["status"] == "logging":
                self.state["status"] = "complete"
            self.state["global_step"] = state.global_step
            _write(self.state)

    trainer_cls = getattr(module, "Trainer", None)
    if trainer_cls is None or getattr(trainer_cls, "_mt1_a10", False):
        return
    orig_init = trainer_cls.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        try:
            self.add_callback(A10Callback())
        except Exception as e:  # noqa: BLE001
            _write({"status": "NOT CHECKABLE", "reason": f"add_callback: {e!r}"})

    trainer_cls.__init__ = __init__
    trainer_cls._mt1_a10 = True


class _PostImportHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "transformers.trainer":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if _LOG:
    sys.meta_path.insert(0, _PostImportHook())
