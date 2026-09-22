"""Synchronous, process-parallel threshold selection for TGDBF."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
from threading import Lock

import numpy as np

from ldpc_common.alist import Alist
from ldpc_py.bin_ldpc_tgdbf import BinLdpcTgdbfDecoder
from visualisation.error_by_energy_visualisation.algorithms import TannerGraph


MAX_BINS = 10_000


@dataclass(frozen=True)
class TuningConfig:
    code: Path
    snr_db: float
    bin_width: float
    train_frames: int = 10_000
    eval_frames: int = 10_000
    seed: int = 42
    workers: int = 1
    iterations: int = 100
    alpha: float = 1.8
    rho: tuple[float, ...] = (0.0,) * 7
    L: int = 7
    output: Path = Path("logs/tgdbf_tune.jsonl")

    def validate(self):
        if not np.isfinite(self.snr_db):
            raise ValueError("SNR must be finite")
        if not np.isfinite(self.bin_width) or self.bin_width <= 0:
            raise ValueError("bin width must be positive and finite")
        if min(self.train_frames, self.eval_frames, self.workers, self.iterations, self.L) <= 0:
            raise ValueError("frame counts, workers, iterations and L must be positive")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if len(self.rho) != self.L or not np.all(np.isfinite(self.rho)):
            raise ValueError("rho must contain L finite values")

    def metadata(self):
        return {
            "code": str(self.code), "snr_db": self.snr_db,
            "objective": "minimize training FER, then BER",
            "bin_width": self.bin_width,
            "train_frames": self.train_frames, "eval_frames": self.eval_frames,
            "seed": self.seed, "workers": self.workers,
            "iterations": self.iterations, "alpha": self.alpha,
            "rho": list(self.rho), "L": self.L, "output": str(self.output),
        }


class RunState:
    """Small thread-safe snapshot read by the web app."""

    def __init__(self, metadata):
        self._lock = Lock()
        self._data = {"status": "starting", "message": "Подготовка данных",
                      "metadata": metadata, "records": []}

    def update(self, **values):
        with self._lock:
            self._data.update(values)

    def append(self, record):
        with self._lock:
            self._data["records"].append(record)

    def snapshot(self):
        with self._lock:
            return {**self._data, "records": list(self._data["records"])}


def _load_graph(code_path):
    code_path = Path(code_path).expanduser().resolve()
    with code_path.open(encoding="utf-8") as handle:
        description = json.load(handle)
    pcm_name = description.get("pcm")
    if not isinstance(pcm_name, str) or not pcm_name:
        raise ValueError("code JSON does not specify pcm")
    pcm = Alist.read(str(code_path.parent / pcm_name)).astype(np.uint8)
    return TannerGraph.from_pcm(pcm), pcm


def _make_received(frames, block_length, snr_db, seed, stream):
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([seed, stream])))
    sigma = 1.0 / np.sqrt(2.0 * 10.0 ** (snr_db / 10.0))
    received = np.ones((frames, block_length), dtype=np.float32)
    received += sigma * rng.standard_normal(received.shape, dtype=np.float32)
    return received


def _partition(array, count):
    return [part for part in np.array_split(array, count) if len(part)]


def _new_decoder(pcm, parameters):
    return BinLdpcTgdbfDecoder(
        None, pcm=pcm, block_length=pcm.shape[1], n_checks=pcm.shape[0],
        n_iterations=1, is_systematic=False, **parameters,
    )


def _new_states(decoder, received, parameters):
    return [decoder.initial_state(word, parameters) for word in received]


def _metrics(decoder, states):
    bit_errors = sum(int(np.count_nonzero(decoder.hard_decision(state) < 0))
                     for state in states)
    frame_errors = sum(bool(np.any(decoder.hard_decision(state) < 0))
                       for state in states)
    return bit_errors, frame_errors


def _probe(decoder, states, active, received, parameters, iteration, width):
    probe_params = {**parameters, "delta": [0.0, 0.0]}
    correct = np.zeros(0, dtype=np.int64)
    incorrect = np.zeros(0, dtype=np.int64)
    frame_events = np.zeros(0, dtype=np.int64)
    active_count = 0
    base_bit_errors, base_frame_errors = _metrics(decoder, states)
    for index in np.flatnonzero(active):
        # Only diagnostics are used: delta=[0,0] would flip minima in the
        # returned state, so that state must never become the current state.
        diagnostics = decoder.step_state(
            states[index], received[index], probe_params, iteration, None,
        ).diagnostics
        relative = diagnostics["margin"]
        scaled = relative / width
        if not np.all(np.isfinite(scaled)) or np.any(scaled >= MAX_BINS):
            raise ValueError(f"More than {MAX_BINS} bins needed; increase --bin-width")
        # Assign each bit to the first candidate whose actual inclusive
        # threshold flips it; this also handles exact bin-edge values.
        cut_count = min(MAX_BINS, int(np.max(scaled)) + 2)
        cutoffs = np.nextafter(width * np.arange(1, cut_count + 1), -np.inf)
        bins = np.searchsorted(cutoffs, relative, side="left")
        if np.any(bins >= MAX_BINS):
            raise ValueError(f"More than {MAX_BINS} bins needed; increase --bin-width")
        correct_bits = decoder.hard_decision(states[index]) < 0
        good = np.bincount(bins[correct_bits]).astype(np.int64)
        bad = np.bincount(bins[~correct_bits]).astype(np.int64)
        size = max(len(correct), len(incorrect), len(good), len(bad))
        correct = np.pad(correct, (0, size - len(correct)))
        incorrect = np.pad(incorrect, (0, size - len(incorrect)))
        frame_events = np.pad(frame_events, (0, size - len(frame_events)))
        correct[:len(good)] += good
        incorrect[:len(bad)] += bad
        after_errors = int(correct_bits.sum()) + np.cumsum(
            np.pad(bad, (0, size - len(bad)))
            - np.pad(good, (0, size - len(good)))
        )
        frame_error_delta = (after_errors > 0).astype(np.int64) - int(correct_bits.any())
        frame_events[:size] += np.diff(np.r_[0, frame_error_delta])
        active_count += 1
    return {
        "correct": correct, "incorrect": incorrect,
        "frame_events": frame_events,
        "base_bit_errors": base_bit_errors,
        "base_frame_errors": base_frame_errors,
        "active_frames": active_count,
    }


def _advance(decoder, states, active, received, parameters, iteration):
    for index in np.flatnonzero(active):
        states[index] = decoder.step_state(
            states[index], received[index], parameters, iteration, None,
        ).fields
        active[index] = not decoder.is_decoded(states[index])
    errors = _metrics(decoder, states)
    return {"bit_errors": errors[0], "frame_errors": errors[1],
            "active_frames": int(active.sum())}


def _worker(connection, pcm, train_received, eval_received, parameters, width):
    try:
        decoder = _new_decoder(pcm, parameters)
        train_states = _new_states(decoder, train_received, parameters)
        eval_states = _new_states(decoder, eval_received, parameters)
        train_active = np.array([not decoder.is_decoded(s) for s in train_states])
        eval_active = np.array([not decoder.is_decoded(s) for s in eval_states])
        train_errors = _metrics(decoder, train_states)
        eval_errors = _metrics(decoder, eval_states)
        connection.send({"train": {"bit_errors": train_errors[0],
                                     "frame_errors": train_errors[1],
                                     "active_frames": int(train_active.sum())},
                         "eval": {"bit_errors": eval_errors[0],
                                    "frame_errors": eval_errors[1],
                                    "active_frames": int(eval_active.sum())}})
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command == "probe":
                connection.send(_probe(decoder, train_states, train_active,
                                       train_received, parameters, value, width))
            elif command == "step":
                iteration, delta = value
                step_params = {**parameters, "delta": [0.0, delta]}
                connection.send({
                    "train": _advance(decoder, train_states, train_active,
                                      train_received, step_params, iteration),
                    "eval": _advance(decoder, eval_states, eval_active,
                                     eval_received, step_params, iteration),
                })
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except Exception as exc:
        try:
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def _request(connections, command, value):
    for connection in connections:
        connection.send((command, value))
    results = [connection.recv() for connection in connections]
    for result in results:
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
    return results


def _sum_metrics(results, cohort, frame_count, block_length):
    bit_errors = sum(item[cohort]["bit_errors"] for item in results)
    frame_errors = sum(item[cohort]["frame_errors"] for item in results)
    return {
        "bit_errors": bit_errors, "frame_errors": frame_errors,
        "ber": bit_errors / (frame_count * block_length),
        "fer": frame_errors / frame_count,
        "active_frames": sum(item[cohort]["active_frames"] for item in results),
    }


def _sum_probes(probes):
    size = max((len(item["correct"]) for item in probes), default=0)
    correct = np.zeros(size, dtype=np.int64)
    incorrect = np.zeros(size, dtype=np.int64)
    frame_events = np.zeros(size, dtype=np.int64)
    for item in probes:
        for key, target in (("correct", correct), ("incorrect", incorrect),
                            ("frame_events", frame_events)):
            values = item[key]
            target[:len(values)] += values
    return {
        "correct": correct, "incorrect": incorrect,
        "frame_events": frame_events,
        "base_bit_errors": sum(item["base_bit_errors"] for item in probes),
        "base_frame_errors": sum(item["base_frame_errors"] for item in probes),
        "active_frames": sum(item["active_frames"] for item in probes),
    }


def _choose_delta(probe, width):
    """Lexicographically minimize training frame errors, then bit errors.

    Every candidate includes bin zero and flips at least one bit. The smallest
    delta wins exact ties. Candidate k flips all bits in bins 0..k.
    """
    correct = probe["correct"]
    incorrect = probe["incorrect"]
    base_frame_errors = probe["base_frame_errors"]
    base_bit_errors = probe["base_bit_errors"]
    frame_errors = base_frame_errors + np.cumsum(probe["frame_events"])
    bit_errors = base_bit_errors + np.cumsum(incorrect - correct)
    if len(correct) == 0:
        raise ValueError("No active energy bins to search")
    best_key = None
    best_bin = 0
    for index, (frames, bits) in enumerate(zip(frame_errors, bit_errors)):
        key = (int(frames), int(bits), index)
        if best_key is None or key < best_key:
            best_key, best_bin = key, index
    # Exclude the next bin's left edge from the selected interval.
    delta = float(np.nextafter((best_bin + 1) * width, -np.inf))
    return {
        "delta": delta, "last_bin": best_bin,
        "correct_flips": int(correct[:best_bin + 1].sum()),
        "incorrect_flips": int(incorrect[:best_bin + 1].sum()),
        "predicted_frame_errors": int(frame_errors[best_bin]),
        "predicted_bit_errors": int(bit_errors[best_bin]),
        "candidate_count": len(correct),
    }


def _write_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def run(config, state):
    """Run tuning; safe to call from a background thread beside Dash."""
    config.validate()
    graph, pcm = _load_graph(config.code)
    train = _make_received(config.train_frames, graph.block_length,
                           config.snr_db, config.seed, 0x54524149)
    evaluation = _make_received(config.eval_frames, graph.block_length,
                                config.snr_db, config.seed, 0x4556414C)
    parameters = {"delta": [0.0, 0.0], "alpha": config.alpha,
                  "rho": list(config.rho), "L": config.L}
    BinLdpcTgdbfDecoder.validate_parameters(parameters)
    parts = min(config.workers, config.train_frames, config.eval_frames)
    train_parts = _partition(train, parts)
    eval_parts = _partition(evaluation, parts)
    context = mp.get_context("spawn")
    processes, connections = [], []
    try:
        state.update(status="running", message=f"Запуск {parts} процессов")
        for train_part, eval_part in zip(train_parts, eval_parts):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(
                child, pcm, train_part, eval_part, parameters, config.bin_width,
            ))
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
        initial = [connection.recv() for connection in connections]
        for result in initial:
            if "error" in result:
                raise RuntimeError(result["error"])
        output = Path(config.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Never silently destroy results from an earlier run.
        with output.open("x", encoding="utf-8") as handle:
            _write_record(handle, {
                "type": "metadata", "started_utc": datetime.now(timezone.utc).isoformat(),
                **config.metadata(), "code": str(Path(config.code).resolve()),
            })
            record = {"type": "iteration", "iteration": 0, "delta": None,
                      "parameters": parameters,
                      "train": _sum_metrics(initial, "train", config.train_frames, graph.block_length),
                      "eval": _sum_metrics(initial, "eval", config.eval_frames, graph.block_length)}
            _write_record(handle, record)
            state.append(record)
            state.update(message="Итерация 0: начальные BER/FER", output=str(output))
            for iteration in range(config.iterations):
                probe = _sum_probes(_request(connections, "probe", iteration))
                if probe["active_frames"] == 0:
                    reason = "На обучающей выборке все слова уже удовлетворяют проверкам"
                    break
                selected = _choose_delta(probe, config.bin_width)
                parts_result = _request(connections, "step", (iteration, selected["delta"]))
                train_metrics = _sum_metrics(parts_result, "train", config.train_frames,
                                             graph.block_length)
                if (train_metrics["frame_errors"] != selected["predicted_frame_errors"]
                        or train_metrics["bit_errors"] != selected["predicted_bit_errors"]):
                    raise RuntimeError("Predicted training BER/FER differs from applied TGDBF step")
                bounds = [0.0, selected["delta"]]
                record = {
                    "type": "iteration", "iteration": iteration + 1,
                    "delta": bounds,
                    "last_bin": selected["last_bin"],
                    "correct_flips_train": selected["correct_flips"],
                    "incorrect_flips_train": selected["incorrect_flips"],
                    "candidate_count": selected["candidate_count"],
                    "bin_correct": probe["correct"].tolist(),
                    "bin_incorrect": probe["incorrect"].tolist(),
                    "active_train_before": probe["active_frames"],
                    "parameters": {**parameters, "delta": bounds},
                    "train": train_metrics,
                    "eval": _sum_metrics(parts_result, "eval", config.eval_frames, graph.block_length),
                }
                _write_record(handle, record)
                state.append(record)
                choice = f"delta={selected['delta']:.6g}"
                state.update(message=f"Итерация {iteration + 1}: {choice}")
                print(f"iteration={iteration + 1} {choice} "
                      f"BER={record['eval']['ber']:.6g} FER={record['eval']['fer']:.6g}", flush=True)
            else:
                reason = f"Достигнут предел {config.iterations} итераций"
            _write_record(handle, {"type": "stop", "reason": reason,
                                   "iteration": state.snapshot()["records"][-1]["iteration"]})
            state.update(status="finished", message=reason)
            print(reason, flush=True)
    finally:
        for connection in connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            connection.close()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
