"""Pin what preregistration requires before the first model call.

Resolves each model's immutable Hugging Face revision SHA into `manifest.json`, records
models that cannot be used, sets the judge identity and generation status, and optionally
stores a live letter-token check. Files whose hashes are frozen in `manifest.file_sha256`
are never written and are re-verified before and after the update.

The judge identity defaults to anthropic / claude-sonnet-4-6 and is overridable with
--judge-provider / --judge-model / --judge-revision. --check-keys reports only whether the
judge API keys resolve from the environment or an upward .env search; key values are never
read into output or written to the manifest.

Usage (PowerShell, from the repo root):
  .venv\\Scripts\\python.exe scripts/preflight.py --models google/gemma-2-2b-it google/gemma-2-9b-it \\
      Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct --generation-status ready --check-keys
  .venv\\Scripts\\python.exe scripts/preflight.py --models google/gemma-2-2b-it \\
      --endpoint https://<workspace>--dgs-vllm-gemma-2-2b-it-serve.modal.run/v1 \\
      --endpoint-model google/gemma-2-2b-it
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KOLKATA = timezone(timedelta(hours=5, minutes=30))
UNRESOLVED = "unresolved_before_generation"
GENERATION_STATUSES = ("not_started", "ready", "in_progress", "complete")
DEFAULT_JUDGE_PROVIDER = "anthropic"
# Sonnet 4.6, not Sonnet 5: Sonnet 5 rejects non-default sampling parameters, and the frozen
# judge configuration mandates temperature 0.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_KEY_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
# Locked files that are committed by hash but deliberately not shipped: the roadmap is the
# authors' private planning document. Absence is a notice; a present copy is hashed as usual.
UNDISTRIBUTED = ("roadmap",)
NOT_DISTRIBUTED = ("NOTICE: locked file %r is not distributed with the repository; its sha256 remains "
                   "in manifest.json (%s) and is verified when the file is present")


class PreflightError(RuntimeError):
    """Raised when the manifest cannot be safely pinned."""


def now_iso() -> str:
    return datetime.now(KOLKATA).isoformat(timespec="seconds")


def dump_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize with the manifest's existing style: 2-space top level, compact inner objects.

    Every top-level key is emitted in place, including ones preflight knows nothing about
    (`holdout_unlock` and anything added later): keys are never dropped, reordered, or
    rewritten, and nested objects keep their own insertion order.
    """
    items = list(manifest.items())
    lines = ["{"]
    for index, (key, value) in enumerate(items):
        rendered = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        comma = "," if index < len(items) - 1 else ""
        lines.append("  %s: %s%s" % (json.dumps(key, ensure_ascii=False), rendered, comma))
    lines.append("}")
    return "\n".join(lines) + "\n"


def locked_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    files = manifest.get("files") or {}
    hashes = manifest.get("file_sha256") or {}
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise PreflightError("manifest files/file_sha256 must be objects")
    return {files[key]: hashes[key] for key in files if key in hashes}


def undistributed_files(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Locked files that may be absent, as relative path -> manifest key."""
    files = manifest.get("files") or {}
    if not isinstance(files, Mapping):
        return {}
    return {files[key]: key for key in UNDISTRIBUTED if key in files}


def check_locked(root: Path, manifest: Mapping[str, Any], stage: str, *, stream=None) -> None:
    """Re-hash every locked file. An undistributed one may be missing: that is reported on
    `stream` (when given) and skipped, but a copy that IS present must still match exactly."""
    optional = undistributed_files(manifest)
    for relative, expected in locked_files(manifest).items():
        path = root / relative
        if relative in optional and not path.is_file():
            if stream is not None:
                print(NOT_DISTRIBUTED % (optional[relative], expected), file=stream)
            continue
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PreflightError("cannot read frozen file %s (%s)" % (relative, stage)) from exc
        if actual != expected:
            raise PreflightError("frozen file changed %s: %s" % (stage, relative))


def find_dotenv(start: Path) -> Path | None:
    """Search `start` and each parent for a .env file (the repo root sits above the worktree)."""
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def key_presence(root: Path, names: Sequence[str] = JUDGE_KEY_NAMES) -> dict[str, str]:
    """Report only where each judge key resolves from; secret values are never returned."""
    result = {name: "absent" for name in names}
    for name in names:
        if (os.environ.get(name) or "").strip():
            result[name] = "environment"
    dotenv = find_dotenv(root)
    if dotenv is None:
        return result
    try:
        lines = dotenv.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key in result and result[key] == "absent" and value.strip().strip("\"'"):
            result[key] = "dotenv"
    return result


class HubResolver:
    """Thin seam over huggingface_hub so tests can inject a fake API."""

    def __init__(self, api: Any = None, token: str | None = None) -> None:
        self._token = token
        if api is not None:
            self._api = api
            return
        try:
            from huggingface_hub import HfApi, get_token
        except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
            raise PreflightError("huggingface_hub is required to resolve revisions") from exc
        self._token = token or get_token()
        self._api = HfApi(token=self._token)

    def sha(self, model_id: str) -> str:
        return self._api.model_info(model_id).sha

    def check_download_access(self, model_id: str, filename: str = "config.json") -> None:
        """Prove the weights are actually reachable, not just the repo card.

        Some gated repos (meta-llama) serve public metadata -- model_info succeeds and returns
        a sha -- while every file HEAD returns 403. Pinning such a revision would hand the
        generation stack a model it cannot load.
        """
        # Distinct hook name so a real HfApi (which has its own get_hf_file_metadata with a
        # keyword-only signature) is never mistaken for an injected test double.
        prober = getattr(self._api, "dgs_probe_file", None)
        if prober is not None:
            prober(model_id, filename)
            return
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        get_hf_file_metadata(url=hf_hub_url(model_id, filename), token=self._token)


def _gated(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("GatedRepoError", "RepositoryNotFoundError"):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in (401, 403)


def resolve_revisions(model_ids: Sequence[str], resolver: HubResolver) -> tuple[dict[str, str], dict[str, str]]:
    """Return (resolved sha per model, unavailable reason per model)."""
    resolved: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    today = datetime.now(KOLKATA).date().isoformat()
    for model_id in model_ids:
        try:
            sha = resolver.sha(model_id)
            resolver.check_download_access(model_id)
        except Exception as exc:  # noqa: BLE001 - classify then re-raise non-access failures
            if _gated(exc):
                unavailable[model_id] = "hf_403_no_license_%s" % today
                continue
            raise PreflightError("cannot resolve revision for %s: %s" % (model_id, exc)) from exc
        if not isinstance(sha, str) or len(sha) != 40 or any(c not in "0123456789abcdefABCDEF" for c in sha):
            raise PreflightError("Hugging Face returned a non-40-hex sha for %s: %r" % (model_id, sha))
        resolved[model_id] = sha.lower()
    return resolved, unavailable


def order_by_model(entries: Mapping[str, Any], canonical: Sequence[str]) -> dict[str, Any]:
    """Emit keys in the manifest's frozen model order so reruns produce identical bytes."""
    known = [key for key in canonical if key in entries]
    return {key: entries[key] for key in known + sorted(set(entries) - set(canonical))}


def apply_preflight(manifest: dict[str, Any], *, resolved: Mapping[str, str],
                    unavailable: Mapping[str, str], judge_provider: str | None,
                    judge_model: str | None, judge_revision: str | None,
                    generation_status: str | None, letter_check: Mapping[str, Any] | None) -> dict[str, Any]:
    """Edit only the preflight-owned fields, leaving all other manifest content untouched."""
    updated = json.loads(json.dumps(manifest))  # deep copy without aliasing frozen views
    models = updated.setdefault("models", {})
    if not isinstance(models, dict):
        raise PreflightError("manifest.models must be an object")

    canonical = list(models.get("ids_in_order") or ())

    def in_canonical_order(entries):
        return order_by_model(entries, canonical)

    revisions = models.get("revisions")
    revisions = dict(revisions) if isinstance(revisions, Mapping) else {}
    revisions.update(resolved)
    models["revisions"] = in_canonical_order(revisions) if revisions else UNRESOLVED

    recorded = models.get("unavailable")
    recorded = dict(recorded) if isinstance(recorded, Mapping) else {}
    recorded.update(unavailable)
    for model_id in resolved:  # a model that now resolves is no longer unavailable
        recorded.pop(model_id, None)
    if recorded:
        models["unavailable"] = in_canonical_order(recorded)
    elif "unavailable" in models:
        del models["unavailable"]

    if judge_provider is not None:
        models["judge_provider"] = judge_provider
    if judge_model is not None:
        models["judge_model"] = judge_model
    if judge_revision is not None:
        models["judge_revision"] = judge_revision
    if generation_status is not None:
        updated["generation_status"] = generation_status
    if letter_check is not None:
        preflight = updated.get("preflight")
        preflight = dict(preflight) if isinstance(preflight, Mapping) else {}
        # Accumulate per model: one endpoint is checked per run, but the manifest has to end
        # up carrying the check for every model that will be generated from.
        checks = preflight.get("letter_token_checks")
        checks = dict(checks) if isinstance(checks, Mapping) else {}
        legacy = preflight.get("letter_token_check")  # single-endpoint layout, pre-2026-08-17
        if isinstance(legacy, Mapping) and legacy.get("model") and legacy["model"] not in checks:
            checks[legacy["model"]] = dict(legacy)
        checks[letter_check["model"]] = dict(letter_check)
        preflight["letter_token_checks"] = in_canonical_order(checks)
        preflight["checked_at"] = now_iso()
        preflight.pop("letter_token_check", None)  # superseded by the per-model mapping
        updated["preflight"] = preflight
    return updated


def summarize(manifest: Mapping[str, Any]) -> str:
    models = manifest.get("models") or {}
    ordered = list(models.get("ids_in_order") or [])
    revisions = models.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    unavailable = models.get("unavailable")
    unavailable = unavailable if isinstance(unavailable, Mapping) else {}
    # Models pinned outside the frozen ids_in_order are post-lock exploratory extensions
    # (configs/models_extension.json); they still have to be visible in this report.
    extension = sorted({key for key in list(revisions) + list(unavailable) if key not in ordered})
    width = max([len(str(item)) for item in ordered + extension] + [len("model")])
    lines = ["%-*s  %-40s  %s" % (width, "model", "revision", "status"),
             "%-*s  %-40s  %s" % (width, "-" * width, "-" * 40, "-" * 12)]
    for model_id in ordered:
        if model_id in revisions:
            lines.append("%-*s  %-40s  %s" % (width, model_id, revisions[model_id], "pinned"))
        elif model_id in unavailable:
            lines.append("%-*s  %-40s  %s" % (width, model_id, unavailable[model_id], "unavailable"))
        else:
            lines.append("%-*s  %-40s  %s" % (width, model_id, UNRESOLVED, "pending"))
    for model_id in extension:
        if model_id in revisions:
            lines.append("%-*s  %-40s  %s" % (width, model_id, revisions[model_id], "pinned (extension)"))
        else:
            lines.append("%-*s  %-40s  %s" % (width, model_id, unavailable[model_id], "unavailable (extension)"))
    ordered = ordered + extension
    lines.append("generation_status: %s" % manifest.get("generation_status"))
    lines.append("judge: %s / %s" % (models.get("judge_provider"), models.get("judge_model")))
    checks = (manifest.get("preflight") or {}).get("letter_token_checks")
    if isinstance(checks, Mapping) and checks:
        lines.append("letter_token_checks:")
        for model_id, check in checks.items():
            results = (check or {}).get("results") or {}
            lines.append("  %-*s  %s" % (width, model_id,
                         ", ".join("%s=%s" % (key, results.get(key)) for key in "ABCD")))
        unchecked = [model_id for model_id in ordered
                     if model_id in revisions and model_id not in checks]
        if unchecked:
            lines.append("  (no letter check yet: %s)" % ", ".join(unchecked))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pin model revisions and judge identity into manifest.json")
    parser.add_argument("--models", nargs="*", default=[], help="Hugging Face model ids to resolve")
    parser.add_argument("--judge-provider", default=DEFAULT_JUDGE_PROVIDER)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-revision")
    parser.add_argument("--check-keys", action="store_true",
                        help="report only whether judge API keys resolve (never prints values)")
    parser.add_argument("--generation-status", choices=GENERATION_STATUSES)
    parser.add_argument("--unavailable", action="append", default=[], metavar="ID=REASON",
                        help="mark a model unusable, e.g. meta-llama/Llama-3.2-3B-Instruct=hf_403_no_license_2026-08-17")
    parser.add_argument("--pin", action="append", default=[], metavar="ID=SHA40",
                        help="pin a revision without contacting Hugging Face, for a model that is "
                             "not a hub repository (Phase 4 serves locally merged DPO weights, whose "
                             "'revision' is the 40-hex prefix of the adapter sha256)")
    parser.add_argument("--endpoint", help="live OpenAI-compatible base URL for the letter-token check")
    parser.add_argument("--endpoint-model", help="served model name at --endpoint (defaults to the first --models entry)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--dry-run", action="store_true", help="print the result without writing manifest.json")
    return parser.parse_args(list(argv) if argv is not None else None)


def run(argv: Sequence[str] | None = None, *, resolver: HubResolver | None = None,
        probe=None, stream=None) -> int:
    args = parse_args(argv)
    out = stream or sys.stdout
    root = Path(args.root).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print("preflight failed: cannot read manifest.json (%s)" % exc, file=out)
        return 1
    if not isinstance(manifest, dict):
        print("preflight failed: manifest.json must be an object", file=out)
        return 1

    explicit: dict[str, str] = {}
    for item in args.unavailable:
        key, separator, reason = str(item).partition("=")
        if not separator or not key.strip() or not reason.strip():
            print("preflight failed: --unavailable expects ID=REASON, got %r" % item, file=out)
            return 1
        explicit[key.strip()] = reason.strip()

    pinned: dict[str, str] = {}
    for item in args.pin:
        key, separator, sha = str(item).partition("=")
        key, sha = key.strip(), sha.strip().lower()
        if not separator or not key or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            print("preflight failed: --pin expects ID=<40-hex sha>, got %r" % item, file=out)
            return 1
        pinned[key] = sha

    try:
        check_locked(root, manifest, "before update", stream=out)
        resolved, unavailable = ({}, {})
        if args.models:
            resolved, unavailable = resolve_revisions(args.models, resolver or HubResolver())
        # Explicit pins win: they name models the hub cannot resolve at all.
        resolved = dict(resolved) | pinned
        unavailable = dict(unavailable) | explicit

        letter_check = None
        if args.endpoint:
            endpoint_model = args.endpoint_model or (args.models[0] if args.models else None)
            if not endpoint_model:
                raise PreflightError("--endpoint requires --endpoint-model or at least one --models entry")
            runner = probe
            if runner is None:
                from src.backend import probe_letter_tokens as runner
            results = runner(args.endpoint, endpoint_model, args.api_key)
            letter_check = {"endpoint": args.endpoint, "model": endpoint_model,
                            "results": {key: bool(results.get(key)) for key in "ABCD"},
                            "all_single_tokens": all(bool(results.get(key)) for key in "ABCD")}

        updated = apply_preflight(manifest, resolved=resolved, unavailable=unavailable,
                                  judge_provider=args.judge_provider, judge_model=args.judge_model,
                                  judge_revision=args.judge_revision,
                                  generation_status=args.generation_status, letter_check=letter_check)
    except PreflightError as exc:
        print("preflight failed: %s" % exc, file=out)
        return 1

    comparable = json.loads(json.dumps(updated))
    baseline = json.loads(json.dumps(manifest))
    for value in (comparable, baseline):
        value.pop("updated_at", None)
        if isinstance(value.get("preflight"), dict):
            value["preflight"].pop("checked_at", None)
    changed = comparable != baseline
    if changed:
        updated["updated_at"] = now_iso()  # idempotent: only bumped when content actually moved

    print(summarize(updated), file=out)
    if args.check_keys:
        found = key_presence(root)
        print("judge api keys: " + ", ".join("%s=%s" % (name, found[name]) for name in JUDGE_KEY_NAMES), file=out)
    if args.dry_run:
        print("dry run: manifest.json not written", file=out)
        return 0
    if not changed:
        print("manifest already pinned; nothing to write", file=out)
        return 0
    manifest_path.write_text(dump_manifest(updated), encoding="utf-8", newline="\n")
    try:
        check_locked(root, updated, "after update")  # silent: the notice was printed above
    except PreflightError as exc:
        print("preflight failed: %s" % exc, file=out)
        return 1
    print("wrote %s" % manifest_path, file=out)
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
