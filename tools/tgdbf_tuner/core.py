"""Synchronous, process-parallel TGDBF threshold tuning and observation."""

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
    ratio: float | None = None
    bin_width: float | None = None
    fixed_delta: float | None = None
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
        if (self.ratio is None) == (self.fixed_delta is None):
            raise ValueError("specify exactly one of ratio or fixed_delta")
        if self.fixed_delta is None:
            if not np.isfinite(self.ratio) or self.ratio <= 0:
                raise ValueError("ratio N must be positive and finite")
            if self.bin_width is None or not np.isfinite(self.bin_width) or self.bin_width <= 0:
                raise ValueError("bin width must be positive and finite")
        elif not np.isfinite(self.fixed_delta) or self.fixed_delta < 0:
            raise ValueError("fixed delta must be finite and non-negative")
        if min(self.train_frames, self.eval_frames, self.workers, self.iterations, self.L) <= 0:
            raise ValueError("frame counts, workers, iterations and L must be positive")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        if len(self.rho) != self.L or not np.all(np.isfinite(self.rho)):
            raise ValueError("rho must contain L finite values")

    def metadata(self):
        return {
            "code": str(self.code), "snr_db": self.snr_db,
            "mode": "fixed" if self.fixed_delta is not None else "ratio",
            "ratio": self.ratio, "bin_width": self.bin_width,
            "fixed_delta": self.fixed_delta,
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
    active_count = 0
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
        bins = np.floor(scaled).astype(np.int64)
        correct_bits = decoder.hard_decision(states[index]) < 0
        good = np.bincount(bins[correct_bits]).astype(np.int64)
        bad = np.bincount(bins[~correct_bits]).astype(np.int64)
        size = max(len(correct), len(incorrect), len(good), len(bad))
        correct = np.pad(correct, (0, size - len(correct)))
        incorrect = np.pad(incorrect, (0, size - len(incorrect)))
        correct[:len(good)] += good
        incorrect[:len(bad)] += bad
        active_count += 1
    return correct, incorrect, active_count


def _advance(decoder, states, active, received, parameters, iteration):
    correct_flips = 0
    incorrect_flips = 0
    for index in np.flatnonzero(active):
        before = decoder.hard_decision(states[index])
        result = decoder.step_state(
            states[index], received[index], parameters, iteration, None,
        )
        flips = result.diagnostics["flip"]
        correct_flips += int(np.count_nonzero(flips & (before < 0)))
        incorrect_flips += int(np.count_nonzero(flips & (before > 0)))
        states[index] = result.fields
        active[index] = not decoder.is_decoded(states[index])
    errors = _metrics(decoder, states)
    return {"bit_errors": errors[0], "frame_errors": errors[1],
            "active_frames": int(active.sum()),
            "correct_flips": correct_flips, "incorrect_flips": incorrect_flips}


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


def _sum_flips(results, cohort):
    correct = sum(item[cohort]["correct_flips"] for item in results)
    incorrect = sum(item[cohort]["incorrect_flips"] for item in results)
    if incorrect:
        ratio, ratio_kind = correct / incorrect, "finite"
    elif correct:
        ratio, ratio_kind = None, "infinite"
    else:
        ratio, ratio_kind = None, "undefined"
    return {"correct": correct, "incorrect": incorrect,
            "ratio": ratio, "ratio_kind": ratio_kind}


def _format_ratio(flips):
    if flips["ratio_kind"] == "infinite":
        return "inf"
    if flips["ratio_kind"] == "undefined":
        return "undefined"
    return f"{flips['ratio']:.6g}"


def _sum_bins(probes):
    size = max((max(len(good), len(bad)) for good, bad, _ in probes), default=0)
    good = np.zeros(size, dtype=np.int64)
    bad = np.zeros(size, dtype=np.int64)
    for local_good, local_bad, _ in probes:
        good[:len(local_good)] += local_good
        bad[:len(local_bad)] += local_bad
    return good, bad, sum(item[2] for item in probes)


def _choose_delta(good, bad, ratio, width):
    """Largest qualifying prefix of bins, starting with E-Emin=0."""
    last = -1
    for index, (correct, incorrect) in enumerate(zip(good, bad)):
        if correct == 0 or (incorrect > 0 and correct < ratio * incorrect):
            break
        last = index
    if last < 0:
        return None
    # Exclude values exactly on the next bin's left edge when possible.
    delta = float(np.nextafter((last + 1) * width, -np.inf))
    return {"delta": delta, "last_bin": last,
            "correct_flips": int(good[:last + 1].sum()),
            "incorrect_flips": int(bad[:last + 1].sum())}


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
    parameters = {"delta": [0.0, config.fixed_delta or 0.0], "alpha": config.alpha,
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
                if config.fixed_delta is not None:
                    last = state.snapshot()["records"][-1]
                    if not (last["train"]["active_frames"] or last["eval"]["active_frames"]):
                        reason = "Все слова удовлетворяют проверкам"
                        break
                    delta = config.fixed_delta
                    parts_result = _request(connections, "step", (iteration, delta))
                    flips_train = _sum_flips(parts_result, "train")
                    flips_eval = _sum_flips(parts_result, "eval")
                    record = {
                        "type": "iteration", "iteration": iteration + 1,
                        "delta": [0.0, delta],
                        "flips_train": flips_train, "flips_eval": flips_eval,
                        "correct_flips_train": flips_train["correct"],
                        "incorrect_flips_train": flips_train["incorrect"],
                        "parameters": parameters,
                        "train": _sum_metrics(parts_result, "train", config.train_frames, graph.block_length),
                        "eval": _sum_metrics(parts_result, "eval", config.eval_frames, graph.block_length),
                    }
                else:
                    probes = _request(connections, "probe", iteration)
                    good, bad, active_train = _sum_bins(probes)
                    if active_train == 0:
                        reason = "На обучающей выборке все слова уже удовлетворяют проверкам"
                        break
                    selected = _choose_delta(good, bad, config.ratio, config.bin_width)
                    if selected is None:
                        reason = "Нет подходящего бина от нуля: нечего флипать"
                        _write_record(handle, {"type": "probe", "iteration": iteration,
                                               "bin_correct": good.tolist(),
                                               "bin_incorrect": bad.tolist(),
                                               "reason": reason})
                        break
                    delta = selected["delta"]
                    parts_result = _request(connections, "step", (iteration, delta))
                    record = {
                        "type": "iteration", "iteration": iteration + 1,
                        "delta": [0.0, delta],
                        "last_bin": selected["last_bin"],
                        "correct_flips_train": selected["correct_flips"],
                        "incorrect_flips_train": selected["incorrect_flips"],
                        "bin_correct": good.tolist(),
                        "bin_incorrect": bad.tolist(),
                        "active_train_before": active_train,
                        "parameters": {**parameters, "delta": [0.0, delta]},
                        "train": _sum_metrics(parts_result, "train", config.train_frames, graph.block_length),
                        "eval": _sum_metrics(parts_result, "eval", config.eval_frames, graph.block_length),
                    }
                _write_record(handle, record)
                state.append(record)
                state.update(message=f"Итерация {iteration + 1}: delta={delta:.6g}")
                ratio_text = ""
                if config.fixed_delta is not None:
                    ratio_text = f" N={_format_ratio(record['flips_train'])}"
                print(f"iteration={iteration + 1} delta={delta:.6g}{ratio_text} "
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
