"""Load an LDPC code and generate a reproducible batch of BPSK/AWGN frames."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from lbc_encoder.lbc_encoder import LBCEncoder
from ldpc_common.alist import Alist

from .algorithms import TannerGraph
from .models import FrameBatch


@dataclass(frozen=True)
class DatasetConfig:
    """Inputs fixed for the lifetime of an interactive session."""

    code_path: Path
    frames: int = 1_000
    snr_db: float = 1.0
    seed: int = 1
    workers: int = 1

    def validate(self):
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if not np.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        if self.workers <= 0:
            raise ValueError("workers must be positive")


@dataclass(frozen=True)
class LoadedDataset:
    graph: TannerGraph
    batch: FrameBatch
    metadata: dict


def load_dataset(config):
    """Load matrices, encode random words when possible, and add AWGN."""
    config.validate()
    code_path = Path(config.code_path).expanduser().resolve()
    with code_path.open("r", encoding="utf-8") as handle:
        code = json.load(handle)

    pcm_path = _resolve_member(code_path, code.get("pcm"), "pcm")
    pcm = Alist.read(str(pcm_path)).astype(np.uint8)
    graph = TannerGraph.from_pcm(pcm)

    generator_name = code.get("generator")
    if generator_name:
        generator_path = _resolve_member(
            code_path,
            generator_name,
            "generator",
        )
        tx_bits = _encode_random_words(
            generator_path,
            frames=config.frames,
            seed=config.seed,
            workers=config.workers,
        )
        source = "random codewords"
    else:
        tx_bits = np.zeros(
            (config.frames, graph.block_length),
            dtype=np.uint8,
        )
        source = "all-zero codewords"

    if tx_bits.shape[1] != graph.block_length:
        raise ValueError("generator and parity-check matrix sizes do not match")

    noise_rng = np.random.Generator(
        np.random.Philox(np.random.SeedSequence([config.seed, 0x4157474E]))
    )
    sigma = 1.0 / np.sqrt(2.0 * 10.0 ** (config.snr_db / 10.0))
    symbols = (1 - 2 * tx_bits.astype(np.int8)).astype(np.int8)
    received = symbols.astype(np.float32)
    received += (
        sigma
        * noise_rng.standard_normal(received.shape, dtype=np.float32)
    )

    received = np.ascontiguousarray(received)
    symbols = np.ascontiguousarray(symbols)
    received.setflags(write=False)
    symbols.setflags(write=False)
    batch = FrameBatch(received=received, transmitted_symbols=symbols)
    batch.validate()
    metadata = {
        "code": code.get("name", code_path.stem),
        "code_path": str(code_path),
        "frames": config.frames,
        "block_length": graph.block_length,
        "checks": graph.n_checks,
        "snr_db": config.snr_db,
        "seed": config.seed,
        "workers": config.workers,
        "word_source": source,
    }
    return LoadedDataset(graph=graph, batch=batch, metadata=metadata)


def _resolve_member(code_path, member, label):
    if not isinstance(member, str) or not member:
        raise ValueError(f"code description does not specify {label}")
    result = code_path.parent / member
    if not result.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {result}")
    return result


def _encode_random_words(generator_path, frames, seed, workers):
    try:
        encoder = LBCEncoder(str(generator_path))
    except OSError as exc:
        raise RuntimeError(
            "the LBC encoder library is not built; run "
            "`python -c \"from lbc_encoder.lbc_encoder import lib_compile; "
            "lib_compile()\"` from the repository root"
        ) from exc
    rng = np.random.Generator(
        np.random.Philox(np.random.SeedSequence([seed, 0x454E434F]))
    )
    information = rng.integers(
        0,
        2,
        size=(frames, encoder.inf_bits),
        dtype=np.uint8,
    )
    codewords = np.empty((frames, encoder.cwd_length), dtype=np.uint8)

    def encode(index):
        encoder.encode(information[index], codewords[index])

    worker_count = min(max(1, int(workers)), frames)
    if worker_count == 1:
        for index in range(frames):
            encode(index)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(encode, range(frames)))
    return codewords
