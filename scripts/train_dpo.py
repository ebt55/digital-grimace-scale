"""Run one Phase 4 DPO arm on Modal and store its training manifest.

Thin CLI around `src.dpo_train_modal.train`: it validates the pair file locally against the
Phase 4 schema (so a malformed record is caught before a GPU is allocated), ships the JSONL
text to the Modal function, and writes the returned manifest to
``results/dpo/train_<arm>.json``.

    C:\\...\\.venv\\Scripts\\python.exe scripts/train_dpo.py --arm A --pairs results/dpo/pairs_A.jsonl
    C:\\...\\.venv\\Scripts\\python.exe scripts/train_dpo.py --arm B --pairs results/dpo/pairs_B.jsonl

Run the arms one at a time: each occupies a full GPU, and the preregistration compares them,
so they must not contend for the same device.

Both arms must be run with the same ``--epochs``; the preregistration requires identical
hyperparameters for A and B, and the manifest records what was actually used.

Two connection workarounds live here because Modal's gRPC hosts are slow to hand-shake from
some networks (see `pin_reachable_modal_addresses` and `relax_modal_channel_timeout`).  If
`Function.remote()` still times out, deploy once with ``--deploy`` and then pass
``--transport https --url <the printed train_web address>``: that route reaches the same
function over ``*.modal.run`` and streams a heartbeat while the GPU works, so a slow link is
never mistaken for a dead job.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dpo_data import ARMS, DpoDataError, validate_pair_record  # noqa: E402

MODAL_GRPC_SUFFIX = ".modal.com"
# Modal's control plane and the region input plane that `Function.remote()` dials.
KNOWN_MODAL_HOSTS = ("api.modal.com", "input-plane.us-east.modal.com")


def _fail(message: str) -> "NoReturn":  # type: ignore[valid-type]
    raise SystemExit("train_dpo: %s" % message)


def pin_reachable_modal_addresses(timeout_s: float = 4.0) -> dict[str, int]:
    """Order DNS answers for Modal's gRPC hosts by measured TCP reachability.

    Modal's load balancers hand out several A records and some of them do not answer from
    this network.  Python's `socket.create_connection` walks the list in order, so every dead
    address costs a full SYN timeout -- long enough that `modal`'s channel-creation deadline
    expires and `Function.remote()` dies with a bare `TimeoutError` while the same host is
    perfectly reachable on another address.  Probing once and putting a live address first
    turns a ~28 s handshake into a fast one.

    This changes nothing outside this process: no hosts file, no system or network setting,
    no other host's resolution.  It is a workaround for a flaky route, not a configuration
    change; see notes/lab-log.md 2026-08-18.
    """
    original = socket.getaddrinfo
    if getattr(original, "_dgs_reachability_ordered", False):
        return {}
    ordered: dict[str, list] = {}
    report: dict[str, int] = {}

    def probe(entry: Sequence[Any]) -> float | None:
        family, _, _, _, address = entry
        started = time.monotonic()
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.settimeout(timeout_s)
        try:
            sock.connect(address)
            return time.monotonic() - started
        except OSError:
            return None
        finally:
            sock.close()

    def patched(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002 - stdlib signature
        entries = original(host, port, family, type, proto, flags)
        if not isinstance(host, str) or not host.endswith(MODAL_GRPC_SUFFIX) or port != 443:
            return entries
        if host not in ordered:
            with ThreadPoolExecutor(max_workers=max(1, len(entries))) as pool:
                timings = list(pool.map(probe, entries))
            live = sorted(((elapsed, index) for index, elapsed in enumerate(timings)
                           if elapsed is not None))
            ordered[host] = [entries[index] for _, index in live] or list(entries)
            report[host] = len(live)
        return ordered[host]

    patched._dgs_reachability_ordered = True  # type: ignore[attr-defined]
    socket.getaddrinfo = patched
    for host in KNOWN_MODAL_HOSTS:  # probe eagerly so the ordering is ready and reportable
        try:
            patched(host, 443, 0, socket.SOCK_STREAM)
        except OSError:
            report[host] = 0
    return report


def load_pairs(path: Path, arm: str) -> tuple[str, int]:
    """Return the verbatim JSONL text plus its record count, after validating every record."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail("cannot read pairs file %s: %s" % (path, exc))
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        _fail("pairs file %s is empty" % path)
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
            validate_pair_record(record)
        except (json.JSONDecodeError, DpoDataError) as exc:
            _fail("invalid pair record at %s:%d: %s" % (path, number, exc))
        if record["arm"] != arm:
            _fail("record %d in %s is arm %r, not the requested arm %r"
                  % (number, path, record["arm"], arm))
    return "\n".join(lines) + "\n", len(lines)


def relax_modal_channel_timeout(attempt_timeout_s: float = 120.0,
                                total_timeout_s: float = 420.0) -> bool:
    """Give Modal's gRPC channel creation longer than its stock 10 s per attempt.

    `modal` decorates `create_channel_with_fallbacks` with `attempt_timeout=10, total_timeout=63`.
    On this machine the TLS handshake to `*.modal.com` currently takes ~28 s (TCP connect is
    instant; the stall is in the handshake, the signature of a path-MTU black hole on the route),
    so every attempt times out and `Function.remote()` fails with a bare `TimeoutError` even
    though the endpoint is perfectly reachable -- as `modal deploy` and `modal app list`, which
    survive on retries, keep demonstrating.  Re-applying the same decorator to the same
    undecorated coroutine with a longer budget changes nothing about the protocol, the
    credentials, or any system setting; it only stops a slow handshake being read as a failure.
    """
    from modal._utils import grpc_utils
    from modal._utils.async_utils import retry

    inner = getattr(grpc_utils.create_channel_with_fallbacks, "__wrapped__", None)
    if inner is None:
        return False
    grpc_utils.create_channel_with_fallbacks = retry(
        n_attempts=6, base_delay=0.5, max_delay=5.0,
        attempt_timeout=attempt_timeout_s, total_timeout=total_timeout_s)(inner)
    return True


REQUIRED_MERGED_FILES = ("config.json", "generation_config.json", "tokenizer_config.json")


def verify_merged(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check the merged directory looks like something vLLM can actually serve.

    A merged arm is only useful to Phase 4 if the arm's directory holds a model config, real
    weight shards, and a tokenizer; a run that trained fine but saved half a model would
    otherwise only be discovered when the evaluation tried to serve it.
    """
    names = list(manifest.get("merged_files") or ())
    weights = [name for name in names if name.endswith(".safetensors")]
    tokenizer = [name for name in names
                 if name in ("tokenizer.json", "tokenizer.model", "spiece.model")]
    missing = [name for name in REQUIRED_MERGED_FILES if name not in names]
    verdict = {
        "path": manifest.get("merged_path"),
        "weight_shards": len(weights),
        "has_index": "model.safetensors.index.json" in names,
        "tokenizer_files": tokenizer,
        "missing_required": missing,
        "servable": bool(weights) and bool(tokenizer) and not missing,
    }
    manifest["merged_verification"] = verdict
    return verdict


def train_over_https(pairs_jsonl_text: str, arm: str, epochs: float, *,
                     url: str | None = None, timeout_s: float = 3 * 60 * 60) -> dict[str, Any]:
    """Drive `dpo_train_modal.train_web` over HTTPS and return its manifest.

    The endpoint answers with newline-delimited JSON: a heartbeat while training runs, then one
    final object.  Deploy the app first (`modal deploy src/dpo_train_modal.py`); the guard token
    is derived from the same local Hugging Face login the deploy baked in, so nothing secret
    travels through the repo or the command line.
    """
    import httpx

    from src.dpo_train_modal import _shared_guard, train_web

    if url:
        endpoint = url
    else:
        # Resolving the URL from the object needs the same gRPC control plane this transport
        # exists to avoid, so a failure here is a prompt for `--url`, not a fatal error.
        try:
            endpoint = train_web.get_web_url()
        except Exception as exc:  # noqa: BLE001 - any hydration failure means "ask for --url"
            _fail("could not look up the train_web URL (%s: %s); pass it with --url, e.g. the "
                  "address `--deploy` printed" % (type(exc).__name__, exc))
    print("train_dpo: HTTPS transport -> %s" % endpoint, flush=True)
    payload = {"token": _shared_guard(), "arm": arm, "pairs_jsonl_text": pairs_jsonl_text,
               "epochs": epochs}
    last: dict[str, Any] = {}
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=120.0)) as client:
        with client.stream("POST", endpoint, json=payload) as response:
            if response.status_code >= 400:
                response.read()
                _fail("train_web returned HTTP %d: %s" % (response.status_code,
                                                          response.text[:400]))
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if last.get("status") == "running":
                    print("train_dpo: ... training %s, %ss elapsed on the container"
                          % (arm, last.get("elapsed_s")), flush=True)
    if "error" in last:
        _fail("train_web reported: %s" % last["error"])
    manifest = last.get("manifest")
    if not isinstance(manifest, dict):
        _fail("train_web returned no manifest (last line: %r)" % (last,))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train_dpo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--pairs", help="pairs JSONL for this arm")
    parser.add_argument("--out", default="results/dpo", help="directory for train_<arm>.json")
    parser.add_argument("--epochs", type=float, default=None,
                        help="override the preregistered 2 epochs (must match across arms)")
    parser.add_argument("--preflight", action="store_true",
                        help="run the CPU-only image/config check instead of training")
    parser.add_argument("--deploy", action="store_true",
                        help="deploy the training app (needed once before --transport https) "
                             "through this script, so the connection workarounds apply")
    parser.add_argument("--transport", choices=("grpc", "https"), default="grpc",
                        help="grpc = Function.remote(); https = the deployed train_web endpoint, "
                             "for when Modal's gRPC invocation path is unreachable")
    parser.add_argument("--url", default=None,
                        help="train_web endpoint URL (default: the deployed app's own URL)")
    args = parser.parse_args(argv)

    reachable = pin_reachable_modal_addresses()
    if reachable:
        print("train_dpo: Modal gRPC hosts, live addresses: %s"
              % ", ".join("%s=%d" % row for row in sorted(reachable.items())), flush=True)

    import modal  # imported here so --help works without the Modal client

    if not relax_modal_channel_timeout():
        print("train_dpo: WARNING: could not extend the Modal channel timeout", file=sys.stderr)

    from src.dpo_train_modal import APP_NAME, NUM_EPOCHS, app, preflight, train, train_web

    if args.deploy:
        from modal.runner import deploy_app

        with modal.enable_output():
            deploy_app(app, name=APP_NAME)
        print("train_dpo: deployed %s; train_web -> %s" % (APP_NAME, train_web.get_web_url()),
              flush=True)
        if not args.arm:
            return 0

    if args.preflight:
        with modal.enable_output(), app.run():
            print(json.dumps(preflight.remote(), indent=2, sort_keys=True))
        return 0

    if not args.arm or not args.pairs:
        _fail("--arm and --pairs are required unless --preflight is given")
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = ROOT / pairs_path
    text, count = load_pairs(pairs_path, args.arm)
    print("train_dpo: arm %s, %d validated pairs from %s" % (args.arm, count, pairs_path),
          flush=True)

    epochs = NUM_EPOCHS if args.epochs is None else float(args.epochs)
    started = time.monotonic()
    if args.transport == "https":
        manifest = train_over_https(text, args.arm, epochs, url=args.url)
    else:
        with modal.enable_output(), app.run():
            manifest = train.remote(text, args.arm, epochs)
    manifest["local_wall_clock_s"] = round(time.monotonic() - started, 1)
    manifest["transport"] = args.transport
    manifest["pairs_path"] = str(pairs_path.relative_to(ROOT))
    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    verdict = verify_merged(manifest)

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    destination = out / ("train_%s.json" % args.arm)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    training = manifest.get("training", {})
    print("train_dpo: arm %s done in %.1f min | %d steps | final loss %s | reward acc %s"
          % (args.arm, manifest["local_wall_clock_s"] / 60.0, training.get("steps"),
             training.get("final_loss"), training.get("final_reward_accuracy")), flush=True)
    print("train_dpo: merged model -> %s (volume %s); adapter sha256 %s"
          % (manifest["merged_path"], manifest["adapter_volume"], manifest["adapter_sha256"]),
          flush=True)
    print("train_dpo: merged directory %s: %d weight shard(s), tokenizer %s, missing %s"
          % ("SERVABLE" if verdict["servable"] else "INCOMPLETE", verdict["weight_shards"],
             verdict["tokenizer_files"] or "NONE", verdict["missing_required"] or "nothing"),
          flush=True)
    print("train_dpo: manifest -> %s" % destination, flush=True)
    return 0 if verdict["servable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
