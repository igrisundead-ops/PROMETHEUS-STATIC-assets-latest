from __future__ import annotations

import atexit
import argparse
import io
import json
import mimetypes
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
RETRYABLE_STATUS_CODES = {402, 429, 500, 502, 503, 504}
REMOVE_BG_URL = "https://api.remove.bg/v1.0/removebg"
LOCKFILE_NAME = ".batch_remove_bg.lock"


@dataclass
class ImageRecord:
    source: Path
    destination: Path | None
    action: str
    reason: str


class CreditsExhaustedError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch background removal for repo assets using Remove.bg."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to scan recursively.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("matting-report.json"),
        help="Path to the JSON report file to write.",
    )
    parser.add_argument(
        "--max-attempt-rounds",
        type=int,
        default=2,
        help="How many full passes across all API keys to allow for retryable failures.",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=300,
        help="Delay between successful requests to avoid hammering the API.",
    )
    return parser.parse_args()


def collect_api_keys() -> list[str]:
    numbered = []
    unnumbered = []
    for name, value in os.environ.items():
        if not value:
            continue
        if name.startswith("REMOVE_BG_KEY_"):
            suffix = name.removeprefix("REMOVE_BG_KEY_")
            if suffix.isdigit():
                numbered.append((int(suffix), value.strip()))
            else:
                unnumbered.append((name, value.strip()))

    numbered.sort(key=lambda item: item[0])
    unnumbered.sort(key=lambda item: item[0])
    keys = [value for _, value in numbered] + [value for _, value in unnumbered]
    return [key for key in keys if key]


def iter_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTS:
            images.append(path)
    return sorted(images)


def has_transparency(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if image.mode in {"RGBA", "LA"}:
                alpha = image.getchannel("A")
                low, _ = alpha.getextrema()
                return low < 255
            if image.mode == "P" and "transparency" in image.info:
                return True
            if "A" in image.getbands():
                alpha = image.getchannel("A")
                low, _ = alpha.getextrema()
                return low < 255
            return False
    except OSError as exc:
        if "truncated" not in str(exc).lower():
            raise
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode in {"RGBA", "LA"}:
                    alpha = image.getchannel("A")
                    low, _ = alpha.getextrema()
                    return low < 255
                if image.mode == "P" and "transparency" in image.info:
                    return True
                if "A" in image.getbands():
                    alpha = image.getchannel("A")
                    low, _ = alpha.getextrema()
                    return low < 255
                return False
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def normalize_image_bytes(source: Path) -> bytes:
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(source) as image:
            image.load()
            normalized = image.convert("RGBA")
            buffer = io.BytesIO()
            normalized.save(buffer, format="PNG")
            return buffer.getvalue()
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def destination_for(source: Path) -> Path:
    return source if source.suffix.lower() == ".png" else source.with_suffix(".png")


def is_git_tracked(root: Path, path: Path) -> bool:
    relative_path = path.relative_to(root).as_posix()
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def remove_with_retry(path: Path, attempts: int = 5, delay_seconds: float = 0.5) -> None:
    for attempt in range(1, attempts + 1):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds * attempt)


def find_existing_replacement(root: Path, source: Path) -> Path | None:
    destination = destination_for(source)
    if not destination.exists():
        return None
    try:
        if has_transparency(destination):
            return destination
    except Exception:
        return None
    return None


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(root: Path) -> Path:
    lock_path = root / LOCKFILE_NAME
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(lock_data.get("pid", 0))
        except Exception:
            pid = 0
        if pid and pid_is_running(pid):
            raise RuntimeError(
                f"Another batch_remove_bg.py process is already running (pid {pid})."
            )
        lock_path.unlink(missing_ok=True)

    payload = json.dumps(
        {
            "pid": os.getpid(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        indent=2,
    )
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    atexit.register(lambda: lock_path.unlink(missing_ok=True))
    return lock_path


def build_report_payload(root: Path, records: list[ImageRecord]) -> dict:
    summary = {
        "total_scanned": len(records),
        "matted": sum(1 for record in records if record.action == "matted"),
        "skipped": sum(1 for record in records if record.action == "skipped"),
        "failed": sum(1 for record in records if record.action == "failed"),
        "converted_to_png": sum(
            1
            for record in records
            if record.action == "matted"
            and record.destination is not None
            and record.source.suffix.lower() in {".jpg", ".jpeg", ".webp"}
            and record.destination.suffix.lower() == ".png"
        ),
    }
    items = [
        {
            "original_path": record.source.relative_to(root).as_posix(),
            "new_path": (
                record.destination.relative_to(root).as_posix()
                if record.destination is not None
                else None
            ),
            "action": record.action,
            "reason": record.reason,
        }
        for record in records
    ]
    return {"summary": summary, "items": items}


def write_report(root: Path, report_path: Path, records: list[ImageRecord]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_report_payload(root, records)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:200] if text else f"HTTP {response.status_code}"

    if isinstance(data, dict):
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                title = first.get("title")
                details = first.get("detail")
                if title and details:
                    return f"{title}: {details}"
                if title:
                    return str(title)
        if "error" in data:
            return str(data["error"])

    return f"HTTP {response.status_code}"


class RemoveBgClient:
    def __init__(self, keys: list[str], pause_ms: int, max_attempt_rounds: int):
        self.keys = keys
        self.pause_ms = pause_ms
        self.max_attempts = max(1, len(keys) * max_attempt_rounds)
        self._key_index = 0
        self.exhausted_keys: set[str] = set()
        self.session = requests.Session()

    def _next_key(self) -> str:
        available = [key for key in self.keys if key not in self.exhausted_keys]
        if not available:
            raise CreditsExhaustedError("All provided Remove.bg keys are out of credits.")
        key = available[self._key_index % len(available)]
        self._key_index += 1
        return key

    def remove_background(self, source: Path) -> tuple[bytes | None, str]:
        last_reason = "No request attempted."
        mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"

        for attempt in range(1, self.max_attempts + 1):
            key = self._next_key()
            try:
                response = self._post_remove_bg(key, source, mime_type=mime_type)
            except requests.RequestException as exc:
                last_reason = f"Request error: {exc}"
                if attempt < self.max_attempts:
                    time.sleep(min(2**attempt, 8))
                    continue
                return None, last_reason

            if response.status_code == 200:
                if self.pause_ms > 0:
                    time.sleep(self.pause_ms / 1000)
                return response.content, "Processed with Remove.bg"

            last_reason = parse_error_message(response)
            if "error reading the image" in last_reason.lower():
                try:
                    normalized_bytes = normalize_image_bytes(source)
                    response = self._post_remove_bg(
                        key,
                        source,
                        mime_type="image/png",
                        file_bytes=normalized_bytes,
                        upload_name=f"{source.stem}-normalized.png",
                    )
                except Exception as exc:
                    last_reason = f"{last_reason} (normalization failed: {exc})"
                else:
                    if response.status_code == 200:
                        if self.pause_ms > 0:
                            time.sleep(self.pause_ms / 1000)
                        return response.content, "Processed with Remove.bg after normalization"
                    last_reason = parse_error_message(response)
            if response.status_code == 402 and "insufficient credits" in last_reason.lower():
                self.exhausted_keys.add(key)
                if len(self.exhausted_keys) == len(self.keys):
                    raise CreditsExhaustedError(
                        "All provided Remove.bg keys are out of credits."
                    )
            if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= self.max_attempts:
                return None, last_reason

            time.sleep(min(2**attempt, 8))

        return None, last_reason

    def _post_remove_bg(
        self,
        key: str,
        source: Path,
        *,
        mime_type: str,
        file_bytes: bytes | None = None,
        upload_name: str | None = None,
    ) -> requests.Response:
        if file_bytes is None:
            with source.open("rb") as image_file:
                return self.session.post(
                    REMOVE_BG_URL,
                    headers={"X-Api-Key": key},
                    files={
                        "image_file": (source.name, image_file, mime_type),
                    },
                    data={
                        "size": "auto",
                        "type": "auto",
                        "format": "png",
                    },
                    timeout=180,
                )

        return self.session.post(
            REMOVE_BG_URL,
            headers={"X-Api-Key": key},
            files={
                "image_file": (upload_name or source.name, file_bytes, mime_type),
            },
            data={
                "size": "auto",
                "type": "auto",
                "format": "png",
            },
            timeout=180,
        )


def write_processed_asset(root: Path, source: Path, output_bytes: bytes) -> Path:
    destination = destination_for(source)
    if destination != source and destination.exists():
        if not is_git_tracked(root, destination):
            try:
                if has_transparency(destination):
                    remove_with_retry(source)
                    return destination
            except Exception:
                pass
        raise FileExistsError(
            f"Destination already exists for extension change: {destination.name}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".png", dir=str(destination.parent)) as tmp:
        tmp.write(output_bytes)
        tmp_path = Path(tmp.name)

    try:
        with Image.open(tmp_path) as processed:
            processed.verify()
        os.replace(tmp_path, destination)
        if destination != source and source.exists():
            remove_with_retry(source)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return destination


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report_path = (root / args.report).resolve()
    try:
        acquire_lock(root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    keys = collect_api_keys()
    if not keys:
        print("No Remove.bg API keys found in environment variables.", file=sys.stderr)
        return 1

    images = iter_images(root)
    records: list[ImageRecord] = []
    client = RemoveBgClient(
        keys=keys,
        pause_ms=max(0, args.pause_ms),
        max_attempt_rounds=max(1, args.max_attempt_rounds),
    )

    print(f"Scanning {len(images)} images under {root}")

    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(root).as_posix()
        print(f"[{index}/{len(images)}] {relative_path}")

        try:
            if not image_path.exists():
                replacement = find_existing_replacement(root, image_path)
                if replacement is not None:
                    records.append(
                        ImageRecord(
                            source=image_path,
                            destination=replacement,
                            action="matted",
                            reason="Source was already replaced before resume.",
                        )
                    )
                else:
                    records.append(
                        ImageRecord(
                            source=image_path,
                            destination=None,
                            action="failed",
                            reason="Source file is missing.",
                        )
                    )
                write_report(root, report_path, records)
                continue

            if has_transparency(image_path):
                records.append(
                    ImageRecord(
                        source=image_path,
                        destination=image_path,
                        action="skipped",
                        reason="Already has transparency.",
                    )
                )
                write_report(root, report_path, records)
                continue
        except Exception as exc:
            print(
                f"Transparency check failed for {relative_path}: {exc}. Attempting API anyway.",
                file=sys.stderr,
            )

        try:
            output_bytes, reason = client.remove_background(image_path)
        except CreditsExhaustedError as exc:
            write_report(root, report_path, records)
            print(str(exc), file=sys.stderr)
            return 2
        if output_bytes is None:
            records.append(
                ImageRecord(
                    source=image_path,
                    destination=None,
                    action="failed",
                    reason=reason,
                )
            )
            write_report(root, report_path, records)
            continue

        try:
            destination = write_processed_asset(root, image_path, output_bytes)
        except Exception as exc:
            records.append(
                ImageRecord(
                    source=image_path,
                    destination=None,
                    action="failed",
                    reason=f"Write failed: {exc}",
                )
            )
            write_report(root, report_path, records)
            continue

        records.append(
            ImageRecord(
                source=image_path,
                destination=destination,
                action="matted",
                reason=reason,
            )
        )
        write_report(root, report_path, records)

    payload = build_report_payload(root, records)
    summary = payload["summary"]
    print(
        "Done. "
        f"Scanned={summary['total_scanned']} "
        f"Matted={summary['matted']} "
        f"Skipped={summary['skipped']} "
        f"Failed={summary['failed']} "
        f"ConvertedToPng={summary['converted_to_png']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
