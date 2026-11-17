from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import re
import logging
import shutil
import sys
import time
import uuid
import warnings
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = [
    "download_url_to_file",
    "get_dir",
    "help",
    "list",
    "load",
    "load_state_dict",
    "load_state_dict_from_url",
    "load_model",
    "set_dir",
    "snapshot_download",
    "list_entrypoints",
]

logger = logging.getLogger(__name__)

# Cache directory resolution: the hub tree lives under $TENSORPLAY_HOME, or
# under $XDG_CACHE_HOME/tensorplay, or ~/.cache/tensorplay when neither is set.
ENV_HOME = "TENSORPLAY_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

VAR_DEPENDENCY = "dependencies"
MODULE_HUBCONF = "hubconf.py"
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
COMMIT_PIN_NAME = ".commit"

# Matches checkpoint URLs pointing directly at a weight file.
_WEIGHT_URL_RE = re.compile(r"^https?://.*\.(pth|pt|ckpt|safetensors|mst)([?#].*)?$", re.I)

# Repositories owned by these accounts are trusted without prompting. Any other
# owner needs explicit acknowledgement on first download.
_TRUSTED_REPO_OWNERS: tuple[str, ...] = ()

_hub_dir: Optional[Path] = None


def get_dir() -> Path:
    """Get the TensorPlay Hub cache directory used for storing downloaded models & weights."""
    if _hub_dir is not None:
        return _hub_dir
    env_home = os.getenv(ENV_HOME)
    if env_home:
        base = Path(env_home).expanduser()
    else:
        xdg = os.getenv(ENV_XDG_CACHE_HOME)
        base = (
            Path(xdg).expanduser() / "tensorplay"
            if xdg
            else Path.home() / ".cache" / "tensorplay"
        )
    return base / "hub"


def set_dir(d: Union[str, Path]) -> None:
    r"""
    Optionally set the TensorPlay Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    if not isinstance(d, (str, Path)):
        raise TypeError(f"Expected directory path to be str or Path, but got {type(d).__name__}.")
    global _hub_dir
    _hub_dir = Path(d).expanduser()

# Download utility
DEFAULT_RETRY_DELAY: float = 1.0
MAX_RETRY_DELAY: float = 10.0
MAX_DOWNLOAD_ATTEMPTS: int = 100
READ_DATA_CHUNK: int = 128 * 1024
USER_AGENT = "TensorPlay/1.0 (Python/{}.{}; {})".format(
    sys.version_info.major,
    sys.version_info.minor,
    sys.platform
)

_PART_SUFFIX = ".part"
_SIDECAR_SUFFIX = ".sha256"
_LOCK_SUFFIX = ".lock"

# Printing a header is part of the visible progress contract.
_PROGRESS_BAR_FORMAT = "    \033[95m{bar:40}\033[0m {n_fmt}/{total_fmt} {rate_fmt} eta {remaining}"


class _RetryableDownloadError(RuntimeError):
    """A transient failure that a later attempt may still recover from."""


def _human_readable_size(size: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _hash_file(path: Path, algo=hashlib.sha256) -> str:
    hasher = algo()
    with open(path, "rb") as f:
        while chunk := f.read(READ_DATA_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _feed_hashers(path: Path, *hashers) -> None:
    """Replay a staged partial file through every incremental hasher."""
    with open(path, "rb") as f:
        while chunk := f.read(READ_DATA_CHUNK):
            for hasher in hashers:
                if hasher is not None:
                    hasher.update(chunk)


def _read_sidecar(sidecar: Path) -> tuple[Optional[str], Optional[int]]:
    """Digest recorded for a finished download, and the size it covers."""
    try:
        tokens = sidecar.read_text().split()
    except OSError:
        return None, None
    if not tokens or len(tokens[0]) != 64:
        return None, None
    size = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else None
    return tokens[0], size


def _write_sidecar(sidecar: Path, digest: str, size: int) -> None:
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    try:
        tmp.write_text(f"{digest}\n{size}\n")
        os.replace(tmp, sidecar)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _matches_recorded_file(
    dst_path: Path, sidecar: Path, hash_prefix: Optional[str], hash_algo
) -> tuple[bool, Optional[str]]:
    """Whether an existing file may be reused, and its full SHA-256 if known.

    A file with no recorded digest is accepted as-is, without a digest for the
    record. When a digest was recorded, its recorded size settles the common
    case without rereading the file; a size mismatch falls back to a full
    reread so a truncated or replaced file is still detected.
    """
    try:
        size = dst_path.stat().st_size
    except OSError:
        return False, None
    digest, recorded_size = _read_sidecar(sidecar)
    if hash_prefix:
        same_size = recorded_size is None or recorded_size == size
        if hash_algo is hashlib.sha256 and digest is not None and same_size:
            return digest.startswith(hash_prefix), digest
        actual = _hash_file(dst_path, hash_algo)
        if not actual.startswith(hash_prefix):
            return False, None
        return True, actual if hash_algo is hashlib.sha256 else None
    if digest is None:
        return True, None
    if recorded_size is not None and recorded_size == size:
        return True, digest
    actual = _hash_file(dst_path, hashlib.sha256)
    if actual == digest:
        return True, actual
    return False, None


@contextlib.contextmanager
def _exclusive_lock(lock_path: Path):
    """Serialize writers of one destination; yields whether the lock is held."""
    try:
        import fcntl
    except ImportError:  # platforms without advisory locks stage privately instead
        yield False
        return
    fd = -1
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        if fd >= 0:
            os.close(fd)
        yield False
        return
    try:
        yield True
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _header_int(headers, name: str) -> Optional[int]:
    value = headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else None


def _content_range_start(headers) -> Optional[int]:
    value = headers.get("Content-Range")
    if not value:
        return None
    match = re.match(r"bytes\s+(\d+)-", value.strip(), re.I)
    return int(match.group(1)) if match else None


def _content_range_total(headers) -> Optional[int]:
    value = headers.get("Content-Range")
    if not value:
        return None
    match = re.match(r"bytes\s+\d+-\d+/(\d+)", value.strip(), re.I)
    return int(match.group(1)) if match else None


def _retry_after(headers) -> Optional[float]:
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(float(value.strip()), 0.0)
    except ValueError:
        return None


def _warn_on_low_disk_space(expected_size: Optional[int], directory: Path) -> None:
    if expected_size is None or expected_size <= 0:
        return
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return
    if free < expected_size:
        warnings.warn(
            "Not enough free disk space to download the file: "
            f"{_human_readable_size(expected_size)} needed, "
            f"{_human_readable_size(free)} available in {directory}.",
            stacklevel=3,
        )


def _start_progress_bar(url: str, total_size: Optional[int], initial: int):
    readable_size = _human_readable_size(total_size) if total_size is not None else "Unknown size"
    print(f"Downloading {url} ({readable_size})")
    if tqdm is None:
        return None
    return tqdm(
        total=total_size,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        bar_format=_PROGRESS_BAR_FORMAT,
        ascii=" ╸━",
        file=sys.stderr,
        leave=True,
    )


def _resume_state(part_path: Path, prefix_algo):
    """Staged byte count plus hashers primed over those bytes."""
    hasher = hashlib.sha256()
    prefix_hasher = prefix_algo() if prefix_algo is not None else None
    size = 0
    if part_path.exists():
        try:
            size = part_path.stat().st_size
            if size:
                _feed_hashers(part_path, hasher, prefix_hasher)
            else:
                part_path.unlink()
        except OSError:
            with contextlib.suppress(OSError):
                part_path.unlink()
            size = 0
            hasher = hashlib.sha256()
            prefix_hasher = prefix_algo() if prefix_algo is not None else None
    return size, hasher, prefix_hasher


def _download_locked(
    *,
    url: str,
    dst_path: Path,
    sidecar: Path,
    locked: bool,
    hash_prefix: Optional[str],
    hash_algo,
    progress: bool,
    timeout: float,
    max_retries: int,
    overwrite: bool,
    allow_resume: bool,
    user_agent: str,
) -> None:
    if dst_path.exists():
        if overwrite:
            dst_path.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
        else:
            accepted, digest = _matches_recorded_file(dst_path, sidecar, hash_prefix, hash_algo)
            if accepted:
                logger.info(f"Target file {dst_path} already matches the recorded digest; skipping download.")
                # Record what a legacy cache entry lacked so later hits stay cheap.
                if digest is not None:
                    size = dst_path.stat().st_size
                    if _read_sidecar(sidecar) != (digest, size):
                        try:
                            _write_sidecar(sidecar, digest, size)
                        except OSError as exc:
                            logger.warning(f"Could not write sha256 sidecar for {dst_path}: {exc}")
                return
            logger.warning(f"Target file {dst_path} does not match the recorded digest. Redownloading.")
            dst_path.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)

    resumable = allow_resume and locked
    if resumable:
        part_path = dst_path.with_name(dst_path.name + _PART_SUFFIX)
    else:
        part_path = dst_path.with_name(f"{dst_path.name}.{uuid.uuid4().hex[:8]}{_PART_SUFFIX}")

    prefix_algo = hash_algo if hash_prefix else None
    attempts_left = max(1, int(max_retries))
    attempts_used = 0
    last_error: Optional[BaseException] = None
    retry_hint: Optional[float] = None
    restarted_after_mismatch = False

    try:
        while attempts_left > 0 and attempts_used < MAX_DOWNLOAD_ATTEMPTS:
            attempts_used += 1
            attempts_left -= 1
            if attempts_used > 1:
                delay = min(DEFAULT_RETRY_DELAY * (2 ** (attempts_used - 2)), MAX_RETRY_DELAY)
                if retry_hint is not None:
                    delay = min(retry_hint, MAX_RETRY_DELAY)
                time.sleep(delay + random.uniform(0.0, 0.1 * delay))
            retry_hint = None

            resume_size, hasher, prefix_hasher = _resume_state(part_path, prefix_algo)
            if resume_size:
                logger.info(f"Resuming {url} at byte {resume_size}")

            headers = {"User-Agent": user_agent, "Accept": "*/*", "Accept-Encoding": "identity"}
            if resume_size:
                headers["Range"] = f"bytes={resume_size}-"
            request = Request(url, headers=headers)
            received = 0

            try:
                with urlopen(request, timeout=timeout) as response:
                    status = getattr(response, "status", None) or 200
                    if status in (200, 206):
                        range_start = _content_range_start(response.headers) if status == 206 else None
                        if status == 206 and range_start is None:
                            # No range advertised: treat the body as continuing at the staged offset.
                            range_start = resume_size
                        if status == 200 or range_start == 0:
                            if resume_size:
                                logger.warning("Server ignored the range request; restarting the download.")
                                part_path.unlink(missing_ok=True)
                                resume_size = 0
                                hasher = hashlib.sha256()
                                prefix_hasher = prefix_algo() if prefix_algo is not None else None
                            expected_size = _header_int(response.headers, "Content-Length")
                        else:
                            if range_start != resume_size:
                                part_path.unlink(missing_ok=True)
                                raise _RetryableDownloadError(
                                    f"server resumed at byte {range_start}, expected {resume_size}"
                                )
                            expected_size = _content_range_total(response.headers)
                            if expected_size is None:
                                length = _header_int(response.headers, "Content-Length")
                                expected_size = resume_size + length if length is not None else None
                    elif status == 404:
                        raise HTTPError(url, status, "File not found", response.headers, None)
                    elif status in (408, 429) or status >= 500:
                        raise HTTPError(url, status, "Server error", response.headers, None)
                    else:
                        raise RuntimeError(f"Unsupported HTTP status code: {status} (URL: {url})")

                    if expected_size is not None:
                        _warn_on_low_disk_space(max(expected_size - resume_size, 0), dst_path.parent)

                    bar = _start_progress_bar(url, expected_size, resume_size) if progress else None
                    try:
                        with open(part_path, "ab" if resume_size else "wb") as handle:
                            try:
                                while True:
                                    buffer = response.read(READ_DATA_CHUNK)
                                    if not buffer:
                                        break
                                    handle.write(buffer)
                                    hasher.update(buffer)
                                    if prefix_hasher is not None:
                                        prefix_hasher.update(buffer)
                                    received += len(buffer)
                                    if bar is not None:
                                        bar.update(len(buffer))
                            finally:
                                handle.flush()
                                os.fsync(handle.fileno())
                    finally:
                        if bar is not None:
                            bar.close()

                    if expected_size is not None and received != expected_size - resume_size:
                        raise _RetryableDownloadError(
                            f"incomplete body: received {received} of {expected_size - resume_size} bytes"
                        )

                    if prefix_hasher is not None:
                        digest = prefix_hasher.hexdigest()
                        if not digest.startswith(hash_prefix):
                            part_path.unlink(missing_ok=True)
                            if resume_size and not restarted_after_mismatch:
                                restarted_after_mismatch = True
                                attempts_left += 1
                                logger.warning("Hash mismatch after resuming; restarting the download from scratch.")
                                continue
                            raise RuntimeError(
                                "Hash check failed!\n"
                                f"File path: {dst_path}\n"
                                f"Expected prefix: {hash_prefix}\n"
                                f"Actual hash: {digest}\n"
                            )

                    os.replace(part_path, dst_path)
                    try:
                        _write_sidecar(sidecar, hasher.hexdigest(), dst_path.stat().st_size)
                    except OSError as exc:
                        logger.warning(f"Could not write sha256 sidecar for {dst_path}: {exc}")
                    return
            except HTTPError as exc:
                last_error = exc
                retry_hint = _retry_after(exc.headers)
                if exc.code == 416:
                    # The staged offset is outside the resource: drop it and
                    # fetch the whole file again.
                    part_path.unlink(missing_ok=True)
                    logger.warning("Server rejected the requested range; restarting the download.")
                elif 400 <= exc.code < 500 and exc.code not in (408, 429):
                    part_path.unlink(missing_ok=True)
                    if exc.code == 404:
                        raise RuntimeError(f"File not found: {url}") from exc
                    raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
                else:
                    logger.warning(f"Download failed (attempt {attempts_used}): HTTP {exc.code}")
            except _RetryableDownloadError as exc:
                last_error = exc
                logger.warning(f"Download failed (attempt {attempts_used}): {exc}")
            except URLError as exc:
                last_error = exc
                logger.warning(f"Download failed (attempt {attempts_used}): {exc}")
            except (TimeoutError, ConnectionError) as exc:
                last_error = exc
                logger.warning(f"Download failed (attempt {attempts_used}): {exc}")

            if received:
                # Data reached the disk, so the endpoint works: keep going while
                # each attempt still advances the staged offset.
                attempts_left = max(attempts_left, 1)
    finally:
        if not resumable:
            part_path.unlink(missing_ok=True)

    message = f"Network error: retried {attempts_used} times, still failed. URL: {url}"
    raise RuntimeError(message) from last_error


def download_url_to_file(
        url: str,
        dst: Union[str, Path],
        hash_prefix: Optional[str] = None,
        progress: bool = True,
        timeout: float = 10.0,
        max_retries: int = 3,
        overwrite: bool = False,
        allow_resume: bool = True,
        user_agent: str = USER_AGENT,
) -> None:
    r"""Download a URL to a local file, atomically and resumably.

    Guarantees:
    - The destination is only ever replaced by an atomically renamed staging
      file, so a concurrent reader never observes a half-written target.
    - An existing file is reused when it matches the digest recorded beside it
      (a size match short-circuits the reread), and replaced otherwise.
    - ``hash_prefix`` is checked while streaming. A mismatch discards the
      staged bytes and, when they came from a resume, retries once from zero.
    - An interrupted transfer resumes with a range request; a server that
      ignores the range restarts the transfer rather than appending a whole
      response to a partial file, and a server that resumes at the wrong offset
      is rejected.
    - A body shorter than the advertised length is treated as a transient
      failure instead of being accepted as complete.
    - Retries are bounded, back off exponentially with jitter, honour
      ``Retry-After``, and do not consume the budget while they keep making
      progress.
    - Writers of one destination are serialized by a lock file; where advisory
      locks are unavailable each writer stages to a private file and renames
      atomically, so a broken lock can cost bandwidth but never corrupt data.

    Args:
        url (str): URL address to download (supports HTTP/HTTPS).
        dst (str | Path): Destination path (including filename) to save the file.
        hash_prefix (Optional[str]): Hexadecimal digest prefix to verify the
            downloaded bytes against, default None.
        progress (bool): Whether to show download progress, default True.
        timeout (float): Per-request network timeout in seconds, default 10.0.
        max_retries (int): Attempt budget for transient failures, default 3.
        overwrite (bool): Whether to replace an existing file, default False.
        allow_resume (bool): Whether an interrupted transfer may resume from
            its staged offset, default True.
        user_agent (str): Custom User-Agent header for HTTP requests.
    """
    if not url.strip():
        raise ValueError("Download URL cannot be an empty string")
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Download URL must use http or https, got {scheme!r}: {url}")
    if hash_prefix is not None:
        if not isinstance(hash_prefix, str) or len(hash_prefix) < 4:
            raise TypeError(
                f"hash_prefix must be a string with length ≥ 4, but got {type(hash_prefix).__name__} "
                f"(length: {len(hash_prefix) if isinstance(hash_prefix, str) else 'N/A'})"
            )
        hash_prefix = hash_prefix.lower()
        if any(ch not in "0123456789abcdef" for ch in hash_prefix):
            raise ValueError(f"hash_prefix must be hexadecimal, got {hash_prefix!r}")

    # A 32-character prefix names an MD5 digest; everything else is SHA-256.
    hash_algo = hashlib.md5 if hash_prefix is not None and len(hash_prefix) == 32 else hashlib.sha256

    dst_path = Path(dst).resolve()
    dst_parent = dst_path.parent
    try:
        dst_parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"Permission denied: cannot create parent directory {dst_parent} - {e}") from e

    sidecar = dst_path.with_name(dst_path.name + _SIDECAR_SUFFIX)
    with _exclusive_lock(dst_path.with_name(dst_path.name + _LOCK_SUFFIX)) as locked:
        _download_locked(
            url=url,
            dst_path=dst_path,
            sidecar=sidecar,
            locked=locked,
            hash_prefix=hash_prefix,
            hash_algo=hash_algo,
            progress=progress,
            timeout=timeout,
            max_retries=max_retries,
            overwrite=overwrite,
            allow_resume=allow_resume,
            user_agent=user_agent,
        )


# ---------------------------------------------------------------------------
# Repository resolution, caching, trust and hubconf loading
# ---------------------------------------------------------------------------


def _parse_repo_info(github: str) -> tuple[str, str, str | None]:
    """Split 'owner/name[:ref]' into its parts, defaulting the ref to main/master."""
    if ":" in github:
        repo_info, ref = github.split(":", 1)
    else:
        repo_info, ref = github, None
    if "/" not in repo_info:
        raise ValueError(f"Invalid repo identifier {github!r}; expected 'owner/name[:ref]'")
    repo_owner, repo_name = repo_info.split("/", 1)

    if ref is None:
        # Default branch: main when it exists, otherwise master.
        try:
            with urlopen(f"https://github.com/{repo_owner}/{repo_name}/tree/main/"):
                ref = "main"
        except HTTPError as e:
            if e.code != 404:
                raise
            ref = "master"
        except URLError as e:
            # Offline: fall back to whatever branch is already in the cache.
            for possible_ref in ("main", "master"):
                if (get_dir() / f"{repo_owner}_{repo_name}_{possible_ref}").exists():
                    ref = possible_ref
                    break
            if ref is None:
                raise RuntimeError(
                    "No internet connection and the repo was not found in the "
                    f"cache ({get_dir()})"
                ) from e
    return repo_owner, repo_name, ref


def _read_url(request: Request) -> str:
    with urlopen(request) as r:
        return r.read().decode(r.headers.get_content_charset("utf-8"))


def _ref_exists_in_repo(repo_owner: str, repo_name: str, ref: str) -> bool:
    # The ref must belong to the repository owner; a fork could otherwise smuggle
    # code in under a trusted owner's ref.
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers["Authorization"] = f"token {token}"
    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags",
    ):
        page = 0
        while True:
            page += 1
            url = Request(f"{url_prefix}?per_page=100&page={page}", headers=headers)
            try:
                response = json.loads(_read_url(url))
            except HTTPError:
                # The token may simply lack permissions; retry without it.
                headers.pop("Authorization", None)
                response = json.loads(_read_url(url))
            if not response:
                break
            for br in response:
                if br["name"] == ref or br["commit"]["sha"].startswith(ref):
                    return True
    return False


def _validate_ref(repo_owner: str, repo_name: str, ref: str) -> None:
    if not _ref_exists_in_repo(repo_owner, repo_name, ref):
        raise ValueError(
            f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
            "If it is a commit from a forked repo, call load() with the fork directly "
            "and pass skip_validation=True."
        )


def _resolve_ref_sha(repo_owner: str, repo_name: str, ref: str) -> Optional[str]:
    """Commit sha the ref currently points at, or None when it cannot be resolved.

    A successful resolution also proves the ref exists in this repository, so
    callers may skip a separate ref-existence check.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers["Authorization"] = f"token {token}"
    url = Request(f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{ref}", headers=headers)
    try:
        data = json.loads(_read_url(url))
    except (HTTPError, URLError, ValueError, OSError):
        return None
    sha = data.get("sha") if isinstance(data, dict) else None
    return sha if isinstance(sha, str) else None


def _check_repo_is_trusted(repo_owner: str, repo_name: str, owner_name_branch: str,
                           trust_repo) -> None:
    hub_dir = get_dir()
    filepath = hub_dir / "trusted_list"

    if not filepath.exists():
        filepath.touch()
    trusted_repos = tuple(line.strip() for line in filepath.read_text().splitlines())

    # Repos already present in the cache count as trusted: they were downloaded
    # on purpose previously.
    trusted_legacy = {d.name for d in hub_dir.iterdir() if d.is_dir()}

    owner_name = f"{repo_owner}_{repo_name}"
    is_trusted = (
        owner_name in trusted_repos
        or owner_name_branch in trusted_legacy
        or repo_owner in _TRUSTED_REPO_OWNERS
    )

    if (trust_repo is False) or (trust_repo == "check" and not is_trusted):
        response = input(
            f"The repository {owner_name} is not on the trusted list and cannot be "
            "downloaded. Do you trust this repository and wish to add it to the "
            "trusted list of repositories (y/N)?"
        )
        if response.lower() not in ("y", "yes"):
            raise RuntimeError("Untrusted repository.")
        is_trusted = True

    if trust_repo is True and not is_trusted:
        with open(filepath, "a") as f:
            f.write(owner_name + "\n")


def _git_archive_link(repo_owner: str, repo_name: str, ref: str) -> str:
    return f"https://github.com/{repo_owner}/{repo_name}/zipball/{ref}"


def _safe_extract_zip(zip_file, extract_to):
    """Extract an archive, rejecting entries that escape the target directory."""
    extract_to = Path(extract_to).resolve(strict=False)

    for member in zip_file.infolist():
        filename = os.path.normpath(member.filename)

        if filename.startswith(("/", "\\")):
            raise ValueError(f"Archive entry has absolute path: {member.filename}")
        if len(filename) >= 2 and filename[1] == ":" and filename[0].isalpha():
            raise ValueError(f"Archive entry has absolute path: {member.filename}")
        if ".." in re.split(r"[/\\]", filename):
            raise ValueError(f"Archive entry contains directory traversal: {member.filename}")

        out = (extract_to / filename).resolve(strict=False)
        if not out.is_relative_to(extract_to):
            raise ValueError(f"Archive entry escapes target directory: {member.filename}")

        zip_file.extract(member, extract_to)


def _get_cache_or_reload(
    github: str,
    force_reload: bool,
    trust_repo,
    verbose: bool = True,
    skip_validation: bool = False,
) -> Path:
    """Download (or reuse) the repo under the hub cache and return its directory.

    The cache directory keeps the ref name for readability, and a ``.commit``
    pin records the exact commit it was fetched at. Unless validation is
    skipped, cache hits re-check the ref's current commit: an unchanged ref is
    reused as-is, a moved ref triggers a refresh, and an unresolvable one
    (offline, rate limited) falls back to using the cache.
    """
    hub_dir = get_dir()
    hub_dir.mkdir(parents=True, exist_ok=True)
    repo_owner, repo_name, ref = _parse_repo_info(github)
    normalized_br = ref.replace("/", "_")
    repo_dir = hub_dir / "_".join([repo_owner, repo_name, normalized_br])
    pin_path = repo_dir / COMMIT_PIN_NAME

    _check_repo_is_trusted(repo_owner, repo_name, repo_dir.name, trust_repo)

    sha = None
    if not skip_validation:
        sha = _resolve_ref_sha(repo_owner, repo_name, ref)

    if (not force_reload) and repo_dir.exists():
        if sha is not None:
            pinned = pin_path.read_text().strip() if pin_path.exists() else ""
            if pinned == sha:
                if verbose:
                    print(f"Using cache found in {repo_dir}", file=sys.stderr)
                return repo_dir
            if not pinned:
                # Cache fetched before pinning existed; adopt the current commit.
                try:
                    pin_path.write_text(sha + "\n")
                except OSError:
                    pass
                if verbose:
                    print(f"Using cache found in {repo_dir}", file=sys.stderr)
                return repo_dir
            if verbose:
                print(f"Refreshing stale cache in {repo_dir} ({pinned[:12]} -> {sha[:12]})", file=sys.stderr)
        else:
            if verbose:
                print(f"Using cache found in {repo_dir}", file=sys.stderr)
            return repo_dir

    if sha is None and not skip_validation:
        _validate_ref(repo_owner, repo_name, ref)

    cached_file = hub_dir / f"{normalized_br}.zip"
    cached_file.unlink(missing_ok=True)

    url = _git_archive_link(repo_owner, repo_name, ref)
    try:
        print(f'Downloading: "{url}" to {cached_file}')
        download_url_to_file(url, str(cached_file), progress=False)
    except HTTPError as err:
        if err.code != 300:
            raise
        # A 300 (Multiple Choices) usually means the ref names both a tag and a
        # branch. Follow git's convention: assume the branch.
        warnings.warn(
            f"The ref {ref} is ambiguous: it may be both a tag and a branch. "
            "Assuming it is a branch; pass refs/heads/<ref> or refs/tags/<ref> "
            "to be explicit (may require skip_validation=True).",
            stacklevel=2,
        )
        url = _git_archive_link(repo_owner, repo_name, f"refs/heads/{ref}")
        download_url_to_file(url, str(cached_file), progress=False)

    with zipfile.ZipFile(cached_file) as zf:
        # The archive's first entry is the top-level directory (typically
        # 'repo-<sha>/'); derive it from the first path component so archives
        # without an explicit directory entry also work.
        extracted_name = zf.infolist()[0].filename.split("/", 1)[0]
        _safe_extract_zip(zf, hub_dir)
    cached_file.unlink(missing_ok=True)

    extracted_repo = hub_dir / extracted_name
    if extracted_repo != repo_dir:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.move(str(extracted_repo), str(repo_dir))

    if sha is not None:
        try:
            pin_path.write_text(sha + "\n")
        except OSError:
            pass
    return repo_dir


# ---------------------------------------------------------------------------
# hubconf discovery, dependency checks and entrypoint loading
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _add_to_sys_path(path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))


def _import_module(name: str, path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise AssertionError(f"failed to load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise AssertionError(f"no loader for {path}")
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Foreign-module compatibility for third-party hubconf files
# ---------------------------------------------------------------------------

# Root-level module names that third-party hubconf files commonly import,
# mapped to the native packages exposing the equivalent API. These strings are
# functional interop identifiers of external ecosystems.
_COMPAT_ROOTS: dict[str, str] = {
    "torch": "tensorplay",
    "torchvision": "tensorplay.vision",
}


def _compat_target_name(name: str) -> Optional[str]:
    """Return the native module name mapped to a foreign module name, or None."""
    for foreign, real in _COMPAT_ROOTS.items():
        if name == foreign or name.startswith(foreign + "."):
            return real + name[len(foreign) :]
    return None


class _AliasLoader:
    """Loader re-exporting an already-initialized module under an alias name."""

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        pass


class _CompatFinder:
    """Meta-path finder resolving foreign module names to native equivalents.

    The aliased module is the real native module object, so classes imported
    through either name share one identity (isinstance and pickle included).
    """

    def __init__(self) -> None:
        # Native modules resolved through this finder during one compat
        # window; used to undo parent-attribute shadowing on exit.
        self.seen: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        import importlib

        mapped = _compat_target_name(fullname)
        if mapped is None or fullname in sys.modules:
            return None
        try:
            module = importlib.import_module(mapped)
        except Exception:
            # Not a native module either: fall through so the interpreter
            # reports ModuleNotFoundError for the originally requested name.
            return None
        self.seen.add(mapped)
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module), origin=module.__file__)


@contextlib.contextmanager
def _foreign_framework_compat():
    """Enable foreign-name resolution while importing/running a hubconf entry.

    Entries already resident in sys.modules bypass meta-path finders, so any
    foreign root loaded earlier in the process would shadow the mapping.  The
    window temporarily evicts those entries; they are restored on exit, after
    alias entries created inside the window are dropped so the module table
    ends up exactly as it entered.
    """
    finder = _CompatFinder()
    saved: dict[str, object] = {}
    for foreign in _COMPAT_ROOTS:
        for name in [k for k in sys.modules if k == foreign or k.startswith(foreign + ".")]:
            saved[name] = sys.modules.pop(name)
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        for foreign in _COMPAT_ROOTS:
            for name in [k for k in sys.modules if k == foreign or k.startswith(foreign + ".")]:
                if name not in saved:
                    del sys.modules[name]
        sys.modules.update(saved)
        _restore_native_exports(finder.seen)


def _restore_native_exports(modules: set[str]) -> None:
    """Restore native package exports shadowed by alias submodule imports.

    Importing a foreign name such as ``torchvision.models.alexnet`` through the
    compat finder aliases the native submodule; the interpreter then binds that
    module onto its native parent package, replacing an exported callable of
    the same name (``tensorplay.vision.models.alexnet``) with the module
    object.  Re-binding the module's own exported value under its own name
    restores the package surface to what its ``__init__`` star-imports set up.
    """
    for mapped in modules:
        module = sys.modules.get(mapped)
        if module is None:
            continue
        child = mapped.rpartition(".")[2]
        parent = sys.modules.get(mapped.rpartition(".")[0])
        if parent is None:
            continue
        value = getattr(module, child, None)
        if value is not None and not isinstance(value, ModuleType):
            setattr(parent, child, value)


def _check_module_exists(name: str) -> bool:
    import importlib.util

    mapped = _compat_target_name(name)
    if mapped is not None:
        try:
            if importlib.util.find_spec(mapped) is not None:
                return True
        except (ImportError, ValueError):
            pass
    return importlib.util.find_spec(name) is not None


def _check_dependencies(m) -> None:
    dependencies = getattr(m, VAR_DEPENDENCY, None)
    if dependencies is not None:
        missing = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if missing:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError("Invalid input: model should be a string of function name")
    _check_dependencies(m)
    fn = getattr(m, model, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"Cannot find callable {model} in hubconf")
    return fn


def _load_hub_module(repo_dir: Path):
    conf = repo_dir / MODULE_HUBCONF
    if not conf.exists():
        raise RuntimeError(f"{repo_dir} has no {MODULE_HUBCONF}")
    with _add_to_sys_path(repo_dir), _foreign_framework_compat():
        return _import_module(MODULE_HUBCONF, str(conf))


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def list(
    github,
    force_reload=False,
    skip_validation=False,
    trust_repo="check",
    verbose=True,
):
    """List all callable entrypoints available in the repo specified by ``github``.

    Args:
        github (str): 'owner/name[:ref]' with an optional ref (tag or branch).
            If the ref is omitted, the default branch is used (``main`` if it
            exists, otherwise ``master``).
        force_reload (bool): discard the cache and force a fresh download.
        skip_validation (bool): skip the check that the ref belongs to the
            repository owner.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``; how to
            treat repositories that are not on the trusted list.
        verbose (bool): mute messages about hitting local caches.
    """
    repo_dir = _get_cache_or_reload(
        github, force_reload, trust_repo, verbose=verbose, skip_validation=skip_validation
    )
    hub_module = _load_hub_module(repo_dir)
    return [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]


def help(github, model, force_reload=False, skip_validation=False, trust_repo="check"):
    """Show the docstring of entrypoint ``model``."""
    repo_dir = _get_cache_or_reload(
        github, force_reload, trust_repo, verbose=True, skip_validation=skip_validation
    )
    hub_module = _load_hub_module(repo_dir)
    entry = _load_entry_from_hubconf(hub_module, model)
    return entry.__doc__


def load(
    repo_or_dir,
    model,
    *args,
    source="github",
    trust_repo="check",
    force_reload=False,
    verbose=True,
    skip_validation=False,
    ref=None,
    **kwargs,
):
    """Load an entrypoint from a GitHub repo or a local directory.

    Args:
        repo_or_dir (str): with ``source='github'``, 'owner/name[:ref]'; with
            ``source='local'``, a path to a directory containing ``hubconf.py``.
        model (str): name of a callable (entrypoint) defined in ``hubconf.py``.
        source (str): 'github' or 'local'.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``; how to
            treat repositories that are not on the trusted list.
        force_reload (bool): discard the cache and force a fresh download
            (no effect with ``source='local'``).
        verbose (bool): mute messages about hitting local caches.
        skip_validation (bool): skip the check that the ref belongs to the
            repository owner.
        ref (str): optional tag or branch; equivalent to appending ``:ref`` to
            ``repo_or_dir``.
    """
    source = source.lower()
    if source not in ("github", "local"):
        raise ValueError(f'Unknown source: "{source}". Allowed values: "github" | "local".')

    if ref is not None:
        if ":" in repo_or_dir:
            raise ValueError("ref must not be given when repo_or_dir already contains ':'")
        repo_or_dir = f"{repo_or_dir}:{ref}"

    if source == "github":
        repo_or_dir = _get_cache_or_reload(
            repo_or_dir,
            force_reload,
            trust_repo,
            verbose=verbose,
            skip_validation=skip_validation,
        )
    return _load_local(repo_or_dir, model, *args, **kwargs)


def _load_local(hubconf_dir, model, *args, **kwargs):
    hub_module = _load_hub_module(Path(hubconf_dir))
    entry = _load_entry_from_hubconf(hub_module, model)
    with _foreign_framework_compat():
        return entry(*args, **kwargs)


def list_entrypoints(repo_id: str, ref: str | None = None, force_reload: bool = False,
                     **kwargs) -> list[str]:
    """Compatibility variant of :func:`list` taking the ref as a separate argument."""
    github = f"{repo_id}:{ref}" if ref else repo_id
    return list(github, force_reload=force_reload, **kwargs)


def load_state_dict_from_url(
    url: str,
    model_dir: Union[str, Path, None] = None,
    map_location=None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = True,
) -> dict:
    """Load a serialized object from the given URL, caching it under ``model_dir``.

    Args:
        url (str): URL of the object to download.
        model_dir (str | Path | None): directory to cache in; defaults to
            ``<get_dir()>/checkpoints``.
        map_location: storage remapping passed to the loader.
        progress (bool): whether to display a progress bar.
        check_hash (bool): require the filename to carry a ``-<sha256-prefix>``
            suffix and verify the file against it.
        file_name (str | None): destination file name; taken from the URL when
            not given.
        weights_only (bool): restrict loading to registered data types; safer
            for untrusted sources (see ``tensorplay.load``).
    """
    if model_dir is None:
        model_dir = get_dir() / "checkpoints"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = file_name or os.path.basename(urlparse(url).path)
    cached_file = model_dir / filename

    hash_prefix = None
    if check_hash:
        match = HASH_REGEX.search(filename) or HASH_REGEX.search(os.path.basename(urlparse(url).path))
        if match is None or not match.group(1):
            raise ValueError(
                f"check_hash=True requires the file name to carry a '-<hash>.' suffix, but {filename!r} has none"
            )
        hash_prefix = match.group(1)
    # Always goes through the downloader: it validates an already cached file
    # against the recorded digest and redownloads a stale or truncated one.
    download_url_to_file(url, str(cached_file), hash_prefix=hash_prefix, progress=progress)

    import tensorplay as tp
    return tp.load(str(cached_file), map_location=map_location, weights_only=weights_only)


# ---------------------------------------------------------------------------
# MEGA backend (megatensors SDK, imported lazily)
# ---------------------------------------------------------------------------

_WEIGHT_EXTS = (".safetensors", ".mst", ".pt", ".pth", ".bin", ".mega")
_MEGA_INDEX_SUFFIX = ".mega.index.json"


def _collect_weight_paths(local_dir: Path) -> list[Path]:
    """Weight entry points inside a downloaded repo snapshot.

    A ``*.mega.index.json`` index alone describes the MEGA shards it references,
    so when one is present it is the only path handed to the loader.
    """
    indexes = sorted(p for p in local_dir.rglob(f"*{_MEGA_INDEX_SUFFIX}") if p.is_file())
    if indexes:
        return indexes
    return sorted(
        p for p in local_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _WEIGHT_EXTS
    )


def _mega_client(endpoint=None, token=None):
    try:
        from megatensors.hub import MegaHubClient
    except ImportError as exc:
        raise ImportError(
            "MEGA hub support requires the 'megatensors' package. "
            "Install it with `pip install megatensors`."
        ) from exc
    return MegaHubClient(endpoint=endpoint, token=token)


def _mega_load_state_dict(paths, device, load_kwargs, trust_policy=None):
    import megatensors

    kwargs = {"framework": "tensorplay", "device": device}
    if trust_policy is not None:
        kwargs["trust_policy"] = trust_policy
    kwargs.update(load_kwargs)
    return megatensors.load_state_dict([str(p) for p in paths], **kwargs)


def snapshot_download(
    repo_id: str,
    *,
    source: str = "mega",
    revision: str = "main",
    include=None,
    exclude=None,
    endpoint=None,
    token=None,
    ref: str | None = None,
) -> Path:
    """Downloads a full repository snapshot (mega or github) into the cache."""
    if source == "mega":
        local_dir = get_dir() / "mega" / repo_id
        client = _mega_client(endpoint, token)
        return client.snapshot_download(
            repo_id, local_dir=local_dir, revision=revision, include=include, exclude=exclude
        )
    elif source == "github":
        github = f"{repo_id}:{ref or revision}" if (ref or revision) else repo_id
        return _get_cache_or_reload(
            github, force_reload=False, trust_repo=True, verbose=False, skip_validation=True
        )
    raise ValueError(f"unknown source '{source}' (expected 'mega' or 'github')")


def load_state_dict(
    repo_or_url: str,
    filename: str | None = None,
    *,
    source: str = "auto",
    device: str = "cpu",
    revision: str = "main",
    endpoint=None,
    token=None,
    ref: str | None = None,
    trust_policy=None,
    **load_kwargs,
) -> dict:
    """Loads weights from MEGA, a GitHub checkpoint URL, or a github repo.

    ``source='auto'`` inspects the argument: an http(s) URL uses the github
    checkpoint path, anything else is treated as a MEGA ``repo_id``.

    ``trust_policy`` (a megatensors ``TrustPolicy``) is MEGA-only: it makes the
    loader verify the signature provenance of every served artifact before any
    tensor is read. GitHub sources have no artifact signing; the trust question
    there is about executing repo code and is governed by ``trust_repo``.
    """
    src = source
    if src == "auto":
        src = "github" if _WEIGHT_URL_RE.match(repo_or_url) else "mega"

    if src == "github":
        if trust_policy is not None:
            raise ValueError("trust_policy applies to MEGA artifacts; GitHub sources are governed by trust_repo")
        if _WEIGHT_URL_RE.match(repo_or_url):
            return load_state_dict_from_url(repo_or_url)
        # github repo holding a bare state-dict entrypoint is rare; route
        # through the entrypoint loader when a filename/entrypoint is given.
        if filename is not None:
            return load(repo_or_url, filename, ref=ref, trust_repo=True, **load_kwargs)
        raise ValueError("github source needs a full checkpoint URL or an entrypoint name")

    client = _mega_client(endpoint, token)
    cache_root = get_dir() / "mega" / repo_or_url
    if filename is not None:
        path = client.download_file(repo_or_url, filename, local_dir=cache_root, revision=revision)
        return _mega_load_state_dict([path], device, load_kwargs, trust_policy=trust_policy)

    local_dir = client.snapshot_download(repo_or_url, local_dir=cache_root, revision=revision)
    paths = _collect_weight_paths(local_dir)
    if not paths:
        raise RuntimeError(f"No weight files found in MEGA repo '{repo_or_url}' (revision={revision})")
    return _mega_load_state_dict(paths, device, load_kwargs, trust_policy=trust_policy)


def load_model(
    repo_or_url: str,
    filename: str | None = None,
    *,
    source: str = "auto",
    device: str = "cpu",
    revision: str = "main",
    endpoint=None,
    token=None,
    ref: str | None = None,
    model=None,
    model_class=None,
    model_kwargs: dict | None = None,
    strict: bool = True,
    assign: bool = False,
    trust_policy=None,
    **load_kwargs,
):
    """Loads weights and returns a ready-to-run model (mega or github).

    Architecture resolution (mega backend): ``model`` instance >
    ``model_class`` callable/dotted-path > repository metadata
    (``model.class`` / ``model.init.*`` via megatensors).

    ``trust_policy`` (a megatensors ``TrustPolicy``) makes the MEGA loader
    verify artifact signature provenance before reading tensors; it is not
    applicable to GitHub sources (see ``load_state_dict``).
    """
    src = source
    if src == "auto":
        src = "github" if (_WEIGHT_URL_RE.match(repo_or_url) or "/" in repo_or_url and filename is None) else "mega"

    if src == "github":
        if trust_policy is not None:
            raise ValueError("trust_policy applies to MEGA artifacts; GitHub sources are governed by trust_repo")
        if model is not None or model_class is not None or model_kwargs is not None:
            raise ValueError("github source resolves architecture via the repo entrypoint")
        entry = filename if filename is not None else repo_or_url.rsplit("/", 1)[-1]
        return load(repo_or_url, entry, ref=ref, trust_repo=True, **load_kwargs)

    client = _mega_client(endpoint, token)
    cache_root = get_dir() / "mega" / repo_or_url
    import megatensors

    if model is not None:
        sd = load_state_dict(
            repo_or_url, filename, source="mega", device=device, revision=revision,
            endpoint=endpoint, token=token, trust_policy=trust_policy, **load_kwargs,
        )
        try:
            model.load_state_dict(sd, strict=strict, assign=assign)
        except TypeError:
            model.load_state_dict(sd, strict=strict)
        if hasattr(model, "to") and device != "cpu":
            model = model.to(device)
        return model

    if filename is not None:
        paths = [client.download_file(repo_or_url, filename, local_dir=cache_root, revision=revision)]
    else:
        local_dir = client.snapshot_download(repo_or_url, local_dir=cache_root, revision=revision)
        paths = _collect_weight_paths(local_dir)
    if not paths:
        raise RuntimeError(f"No weight files found in MEGA repo '{repo_or_url}' (revision={revision})")

    return megatensors.load_model(
        [str(p) for p in paths],
        device=device,
        framework="tensorplay",
        model_class=model_class,
        model_kwargs=model_kwargs,
        strict=strict,
        assign=assign,
        trust_policy=trust_policy,
        **load_kwargs,
    )
