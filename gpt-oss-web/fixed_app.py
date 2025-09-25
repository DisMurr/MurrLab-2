#!/usr/bin/env python3
"""
GPT-OSS 20B Fixed Interface
Based on the working memory-optimized approach
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr
import gc
import sys
import os
import json
import time
import threading
import logging
from logging.handlers import RotatingFileHandler
import hashlib
from typing import Optional
import shutil
import sqlite3

try:
    import psutil  # CPU/RAM/disk metrics
except Exception:  # type: ignore
    psutil = None

try:
    import pynvml  # GPU utilization (optional)
except Exception:  # type: ignore
    pynvml = None

try:
    import requests
except Exception:
    requests = None  # we'll handle absence gracefully

class FixedGPTInterface:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "../gpt-oss/gpt-oss-20b"
        self.loading = False
        # Backend API settings
        self.use_backend: bool = True
        self.backend_url: str = "http://127.0.0.1:8000"
        # HTTP session reuse for efficiency
        self.http = requests.Session() if requests is not None else None
        # Concurrency, dedupe, and safety
        self._lock = threading.Lock()
        self._inflight = 0
        self._last_msg_hash: Optional[str] = None
        self._last_msg_time: float = 0.0
        self._cb_failures: int = 0
        self._cb_open_until: float = 0.0
        # Health watcher state
        self._health = {
            "last_ok": 0.0,
            "last_check": 0.0,
            "latency_ms": None,
            "http_code": None,
            "consecutive_failures": 0,
            "status": "unknown",
        }
        self._watcher_started = False
        # Logging setup
        self._logger = self._init_logging()
        # Start health watcher in background
        self._start_health_watcher()
        # Ops: metrics and cancel
        self._metrics = {
            "total_requests": 0,
            "total_success": 0,
            "total_errors": 0,
            "last_latency_ms": None,
            "avg_latency_ms": None,
            "last_backend_code": None,
            "last_output_chars": 0,
            "last_error": "",
        }
        self._cancel_requested = False
        self._current_stream_resp = None
        # Telemetry DB
        self._db_path = os.path.join(os.path.dirname(__file__), "telemetry.db")
        self._init_db()

    def _init_logging(self) -> logging.Logger:
        logger = logging.getLogger("gpt_oss_frontend")
        logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if reloaded
        if not logger.handlers:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "frontend.log")
            handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
            fmt = logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(fmt)
            logger.addHandler(handler)
            # Also echo warnings+ to stderr for visibility
            stderr = logging.StreamHandler()
            stderr.setLevel(logging.WARNING)
            stderr.setFormatter(fmt)
            logger.addHandler(stderr)
        return logger

    def _start_health_watcher(self):
        if self._watcher_started or requests is None:
            return
        self._watcher_started = True
        t = threading.Thread(target=self._health_loop, name="backend-health", daemon=True)
        t.start()

    def _health_loop(self):
        interval = 10.0
        while True:
            try:
                if not self.use_backend:
                    time.sleep(interval)
                    continue
                url = self._normalize_backend_url(self.backend_url)
                t0 = time.time()
                ok = False
                code = None
                try:
                    r = self.http.get(url + "/", timeout=5) if self.http else requests.get(url + "/", timeout=5)
                    code = r.status_code
                    ok = r.ok
                except Exception:
                    ok = False
                lat = int((time.time() - t0) * 1000)
                self._health["last_check"] = time.time()
                self._health["latency_ms"] = lat
                self._health["http_code"] = code
                if ok:
                    self._health["last_ok"] = time.time()
                    self._health["consecutive_failures"] = 0
                    self._health["status"] = "healthy"
                else:
                    self._health["consecutive_failures"] += 1
                    self._health["status"] = "unreachable"
                # basic circuit: if 3+ failures, open breaker for 20s
                if not ok:
                    self._cb_failures += 1
                    if self._cb_failures >= 3 and time.time() < self._cb_open_until - 1:
                        # breaker already open; extend slightly
                        self._cb_open_until = max(self._cb_open_until, time.time() + 5)
                    elif self._cb_failures >= 3:
                        self._cb_open_until = time.time() + 20
                        self._logger.warning("Circuit opened for 20s due to repeated health check failures")
                else:
                    self._cb_failures = 0
                time.sleep(interval)
            except Exception as e:
                # Never crash the watcher
                self._logger.exception(f"Health watcher error: {e}")
                time.sleep(interval)

    # --- Telemetry DB ---
    def _init_db(self):
        try:
            con = sqlite3.connect(self._db_path)
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL, name TEXT, action TEXT, status TEXT, duration_ms INTEGER, details TEXT)")
            cur.execute("CREATE TABLE IF NOT EXISTS errors (id INTEGER PRIMARY KEY, ts REAL, source TEXT, message TEXT, traceback TEXT, context TEXT)")
            cur.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, ts REAL, route TEXT, http_code INTEGER, latency_ms INTEGER, payload_len INTEGER, response_len INTEGER)")
            con.commit(); con.close()
        except Exception as e:
            self._logger.warning(f"Telemetry DB init failed: {e}")

    def _db_insert(self, sql: str, params: tuple):
        try:
            con = sqlite3.connect(self._db_path)
            cur = con.cursor()
            cur.execute(sql, params)
            con.commit(); con.close()
        except Exception as e:
            self._logger.warning(f"Telemetry DB insert failed: {e}")

    def db_log_event(self, name: str, action: str, status: str, duration_ms: int = 0, details: str = ""):
        self._db_insert("INSERT INTO events(ts,name,action,status,duration_ms,details) VALUES(?,?,?,?,?,?)",
                        (time.time(), name, action, status, duration_ms, details))

    def db_log_error(self, source: str, message: str, traceback_text: str = "", context: str = ""):
        self._db_insert("INSERT INTO errors(ts,source,message,traceback,context) VALUES(?,?,?,?,?)",
                        (time.time(), source, message, traceback_text, context))

    def db_log_request(self, route: str, http_code: Optional[int], latency_ms: int, payload_len: int = 0, response_len: int = 0):
        self._db_insert("INSERT INTO requests(ts,route,http_code,latency_ms,payload_len,response_len) VALUES(?,?,?,?,?,?)",
                        (time.time(), route, http_code if http_code is not None else -1, latency_ms, payload_len, response_len))

    def db_recent(self, table: str, limit: int = 50) -> str:
        try:
            con = sqlite3.connect(self._db_path)
            cur = con.cursor()
            if table not in ("events", "errors", "requests"):
                return "Invalid table"
            cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall(); con.close()
            out = []
            for r in rows:
                out.append(str(r))
            return "\n".join(out) if out else f"(no {table})"
        except Exception as e:
            return f"DB read failed: {e}"

    # --- Backend telemetry ---
    def send_telemetry(self, level: str, name: str, details: dict):
        if requests is None:
            return
        try:
            url = self._normalize_backend_url(self.backend_url) + "/v1/telemetry"
            payload = {"ts": time.time(), "level": level, "name": name, "details": details}
            (self.http or requests).post(url, json=payload, timeout=2)
        except Exception:
            pass

    # --- Event guard ---
    def run_guarded_event(self, name: str, gen, expected_max_seconds: float = 60.0):
        t0 = time.time(); yielded = False
        self.db_log_event(name, action="start", status="running")
        self.send_telemetry("info", f"event:{name}", {"status": "start"})
        try:
            for chunk in gen:
                yielded = True
                # progress heartbeat: if too long since start, still ok because we yield
                dt = time.time() - t0
                if dt > expected_max_seconds:
                    self.db_log_error(name, f"Timeout after {int(dt)}s", context="guard")
                    self.send_telemetry("error", f"event:{name}", {"error": "timeout", "seconds": dt})
                    break
                yield chunk
            status = "ok" if yielded else "no_output"
            dt_ms = int((time.time() - t0) * 1000)
            self.db_log_event(name, action="end", status=status, duration_ms=dt_ms)
            self.send_telemetry("info", f"event:{name}", {"status": status, "duration_ms": dt_ms})
        except Exception as e:
            dt_ms = int((time.time() - t0) * 1000)
            self.db_log_event(name, action="end", status="error", duration_ms=dt_ms, details=str(e))
            self.db_log_error(name, str(e))
            self.send_telemetry("error", f"event:{name}", {"exception": str(e)})
            raise

    def health_summary(self) -> str:
        h = self._health
        lines = [
            f"Backend mode: {self.use_backend}",
            f"Backend URL: {self._normalize_backend_url(self.backend_url)}" if self.use_backend else "Backend URL: (n/a in local mode)",
            f"Status: {h.get('status')}",
            f"Last check: {int(time.time()-h.get('last_check',0))}s ago" if h.get('last_check') else "Last check: never",
            f"Last OK: {int(time.time()-h.get('last_ok',0))}s ago" if h.get('last_ok') else "Last OK: never",
            f"Latency: {h.get('latency_ms')} ms" if h.get('latency_ms') is not None else "Latency: n/a",
            f"HTTP: {h.get('http_code')}",
            f"Consecutive failures: {h.get('consecutive_failures')}",
        ]
        # Circuit breaker status
        if time.time() < self._cb_open_until:
            lines.append(f"Circuit breaker: OPEN for {int(self._cb_open_until - time.time())}s")
        else:
            lines.append("Circuit breaker: closed")
        # Inflight
        lines.append(f"Inflight requests: {self._inflight}")
        return "\n".join(lines)

    def recent_logs(self, max_lines: int = 80) -> str:
        try:
            log_path = os.path.join(os.path.dirname(__file__), "logs", "frontend.log")
            if not os.path.exists(log_path):
                return "(no logs yet)"
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:]) if lines else "(empty log)"
        except Exception as e:
            return f"Failed to read logs: {e}"

    def recent_logs_merged(self, include_backend: bool = True, max_lines: int = 200) -> str:
        try:
            entries = []
            # Frontend
            fpath = os.path.join(os.path.dirname(__file__), "logs", "frontend.log")
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f.readlines()[-max_lines:]:
                        entries.append(("frontend", line.rstrip()))
            # Backend
            if include_backend:
                bpath = os.path.join(os.path.dirname(__file__), "backend.log")
                if os.path.exists(bpath):
                    with open(bpath, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f.readlines()[-max_lines:]:
                            entries.append(("backend", line.rstrip()))
            if not entries:
                return "(no log entries yet)"
            # Simple merge: keep order per-file tail, then combine with prefixes
            return "\n".join([f"[{src}] {txt}" for src, txt in entries][-max_lines:])
        except Exception as e:
            return f"Failed to read logs: {e}"

    def system_usage_summary(self) -> str:
        lines = []
        # Time
        lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))
        # CUDA/GPU
        cuda = torch.cuda.is_available()
        lines.append(f"CUDA available: {cuda}")
        if cuda:
            try:
                n = torch.cuda.device_count()
                for i in range(n):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory // (1024**3)
                    allocated = torch.cuda.memory_allocated(i) // (1024**3)
                    reserved = torch.cuda.memory_reserved(i) // (1024**3)
                    approx_free = max(0, total - reserved)
                    lines.append(f"GPU{i}: {name} | total={total} GiB, allocated(this proc)={allocated} GiB, reserved(this proc)={reserved} GiB, ~free={approx_free} GiB")
                # Optional NVML utilization
                if pynvml is not None:
                    try:
                        pynvml.nvmlInit()
                        for i in range(n):
                            h = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(h)
                            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                            lines.append(f"GPU{i} util: core={util.gpu}% mem={util.memory}% | mem_used={mem.used//(1024**2)} MiB")
                    except Exception:
                        pass
                else:
                    lines.append("GPU util: pynvml not installed (pip install nvidia-ml-py)")
            except Exception:
                pass
        # CPU/RAM
        if psutil is not None:
            try:
                cpu = psutil.cpu_percent(interval=0.0)
                vm = psutil.virtual_memory()
                lines.append(f"CPU: {cpu}% | RAM: {vm.used//(1024**3)}/{vm.total//(1024**3)} GiB ({vm.percent}%)")
            except Exception:
                pass
        else:
            lines.append("CPU/RAM: psutil not installed")
        # Disk (offload folder)
        try:
            offload_dir = os.path.join(os.path.dirname(__file__), "offload")
            os.makedirs(offload_dir, exist_ok=True)
            try:
                if psutil is not None:
                    du = psutil.disk_usage(offload_dir)
                    lines.append(f"Disk(offload): {du.used//(1024**3)}/{du.total//(1024**3)} GiB ({du.percent}%)")
                else:
                    tu = shutil.disk_usage(offload_dir)
                    used = (tu.total - tu.free)//(1024**3)
                    lines.append(f"Disk(offload): {used}/{tu.total//(1024**3)} GiB")
            except Exception:
                pass
        except Exception:
            pass
        return "\n".join(lines)

    def metrics_snapshot(self) -> str:
        m = self._metrics
        parts = [
            f"Total requests: {m['total_requests']}",
            f"Total success: {m['total_success']}",
            f"Total errors: {m['total_errors']}",
            f"Last latency: {m['last_latency_ms']} ms" if m['last_latency_ms'] is not None else "Last latency: n/a",
            f"Avg latency: {int(m['avg_latency_ms'])} ms" if m['avg_latency_ms'] is not None else "Avg latency: n/a",
            f"Last backend HTTP: {m['last_backend_code']}",
            f"Last output chars: {m['last_output_chars']}",
            (f"Last error: {m['last_error'][:160]}" if m['last_error'] else "Last error: -"),
        ]
        return "\n".join(parts)

    def monitor_snapshot(self) -> str:
        return (
            "=== Metrics ===\n" + self.metrics_snapshot() + "\n\n" +
            "=== Health ===\n" + self.health_summary()
        )

    def _record_metrics(self, success: bool, latency_ms: int, http_code: Optional[int] = None, out_text: Optional[str] = None, error: Optional[str] = None):
        m = self._metrics
        if success:
            m["total_success"] += 1
        else:
            m["total_errors"] += 1
        m["last_latency_ms"] = latency_ms
        if m["avg_latency_ms"] is None:
            m["avg_latency_ms"] = latency_ms
        else:
            m["avg_latency_ms"] = 0.8 * m["avg_latency_ms"] + 0.2 * latency_ms
        if http_code is not None:
            m["last_backend_code"] = http_code
        if out_text is not None:
            m["last_output_chars"] = len(out_text)
        if error:
            m["last_error"] = str(error)

    def stop_generation(self) -> str:
        self._cancel_requested = True
        try:
            if self._current_stream_resp is not None:
                try:
                    self._current_stream_resp.close()
                except Exception:
                    pass
                finally:
                    self._current_stream_resp = None
        finally:
            return "⛔ Stop requested. Current stream will halt shortly."

    def _hash_message(self, message: str) -> str:
        return hashlib.sha1(message.encode("utf-8")).hexdigest()

    def _normalize_backend_url(self, url: str) -> str:
        if not url:
            return self.backend_url
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        return url.rstrip("/")
    
    def clear_memory(self):
        """Clear GPU memory thoroughly"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def device_summary(self) -> str:
        """Summarize where the model is placed (GPU/CPU) and offload folder."""
        lines = []
        if self.use_backend:
            lines.append(f"Mode: Backend API ({self.backend_url})")
        else:
            lines.append(f"Mode: Local Transformers")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                total = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                reserved = torch.cuda.memory_reserved(0) // (1024**3)
                allocated = torch.cuda.memory_allocated(0) // (1024**3)
                lines.append(f"GPU: {torch.cuda.get_device_name(0)} ({total} GiB total, {allocated} GiB allocated, {reserved} GiB reserved)")
            except Exception:
                pass
        if self.use_backend:
            lines.append("Backend-managed model; local model not loaded")
            return "\n".join(lines)
        if self.model is None:
            lines.append("Model: not loaded")
            return "\n".join(lines)
        # Try to read device map
        device_map = getattr(self.model, 'hf_device_map', None)
        if device_map:
            gpu_count = sum(1 for v in device_map.values() if isinstance(v, str) and v.startswith('cuda'))
            cpu_count = sum(1 for v in device_map.values() if v == 'cpu')
            lines.append(f"Modules on GPU: {gpu_count}, on CPU: {cpu_count}")
        else:
            # Fallback: sample a few parameters
            try:
                any_param = next(self.model.parameters())
                lines.append(f"Model first param device: {any_param.device}")
            except Exception:
                pass
        offload_dir = os.path.join(os.path.dirname(__file__), "offload")
        lines.append(f"Offload folder: {offload_dir}")
        return "\n".join(lines)
    
    def load_model(self, gpu_mem_gib: int = 12, gpu_only: bool = True):
        """Load model with MXFP4 if available.
        - If gpu_only is True: force all weights to GPU (no offload, no CPU fallback).
        - Else: Use GPU+CPU offload with max_memory caps to avoid OOM and fall back to CPU if needed.
        """
        if self.use_backend:
            # Do not hit the network here—just set status for clarity.
            url = self._normalize_backend_url(self.backend_url)
            yield f"ℹ️ Using Backend API at {url}. Click 'Test Backend' to verify, or just Send a message."
            return
        if self.loading:
            return "⏳ Model is already loading..."
        
        if self.model is not None:
            return "✅ Model is already loaded!"
        
        self.loading = True
        
        try:
            # Clear memory first
            self.clear_memory()
            
            # Load tokenizer (local mode only)
            yield "📥 Loading tokenizer (local mode)... this can take 10-60s on first run"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure allocator
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            # Save last requested settings
            self._last_gpu_mem_gib = gpu_mem_gib
            self._last_gpu_only = gpu_only

            if gpu_only:
                # Strict GPU-only path: everything on cuda:0, no offload, no fallback
                if not torch.cuda.is_available():
                    yield "❌ GPU-only requested but no CUDA device is available."
                    return
                self.clear_memory()
                yield "🤖 Loading on GPU only (4-bit quantized). Estimated 2-5 minutes on first load..."
                try:
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        quantization_config=bnb_cfg,
                        device_map={"": "cuda:0"},
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                    )
                except Exception as e_q:
                    # Fallback: try non-quantized (may OOM)
                    yield f"⚠️ 4-bit quantized load failed: {e_q}. Trying non-quantized GPU load (may OOM)..."
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map={"": "cuda:0"},
                        trust_remote_code=True,
                        dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                    )
            else:
                # GPU+CPU offload path with disk spill to avoid OOM
                offload_dir = os.path.join(os.path.dirname(__file__), "offload")
                os.makedirs(offload_dir, exist_ok=True)
                self.clear_memory()
                yield f"🤖 Loading with GPU+CPU offload (GPU cap ~{gpu_mem_gib} GiB). Estimated 2-5 minutes on first load..."
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                        offload_folder=offload_dir,
                        offload_state_dict=True,
                        max_memory={
                            0: f"{int(gpu_mem_gib)}GiB",
                            "cpu": "48GiB",
                        },
                    )
                except Exception as e1:
                    # Fallback: CPU-only model load (slow but won’t OOM)
                    yield f"⚠️ GPU+CPU offload failed: {e1}\nTrying CPU-only load (slow)..."
                    self.clear_memory()
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="cpu",
                        trust_remote_code=True,
                        dtype=torch.float32,
                        low_cpu_mem_usage=True,
                    )

            # Ensure eval mode for inference
            try:
                self.model.eval()
            except Exception:
                pass
            
            self.loading = False
            yield "✅ Model loaded successfully! Ready to chat."
            
        except Exception as e:
            self.loading = False
            self.clear_memory()
            error_msg = f"❌ Failed to load model: {str(e)}"
            yield error_msg
    
    def chat(self, message, max_tokens=64, temperature=1.0, top_p=0.9, stream=False):
        """Generate chat response using backend API or local model depending on mode"""
        now = time.time()
        # Duplicate suppression: ignore exact same message within 1.5s window
        h = self._hash_message(message or "")
        if self._last_msg_hash == h and (now - self._last_msg_time) < 1.5:
            return "🛑 Duplicate message ignored (sent too quickly)."
        self._last_msg_hash = h
        self._last_msg_time = now
        # Prevent overlapping requests
        if self._inflight > 0:
            return "⏳ Another request is in progress. Please wait..."
        self._inflight += 1
        req_id = f"req-{int(now*1000)}"
        t_start = time.time()
        self._logger.info(f"{req_id} start mode={'backend' if self.use_backend else 'local'} stream={stream}")
        # metrics init
        self._metrics["total_requests"] += 1
        success_flag = False
        http_code_local = None
        out_text_local = None
        error_msg = None
        self._cancel_requested = False
        try:
            if self.use_backend:
                if requests is None:
                    return "❌ Python 'requests' package is not installed. Please install it (pip install requests) or switch to Local mode."
                # Circuit breaker
                if time.time() < self._cb_open_until:
                    cooldown = int(self._cb_open_until - time.time())
                    return f"⛔ Backend temporarily unavailable (circuit breaker). Retry in {cooldown}s."
                try:
                    url = self._normalize_backend_url(self.backend_url) + "/v1/responses"
                    payload = {
                        "input": message,
                        "temperature": float(temperature),
                        "max_output_tokens": int(max_tokens),
                        "stream": bool(stream),
                        "metadata": {"__debug": True},
                    }
                    sess = self.http or requests
                    if stream:
                        # SSE streaming
                        r = sess.post(url, json=payload, stream=True, timeout=(10, 180))
                        self._current_stream_resp = r
                        if not r.ok:
                            self._cb_failures += 1
                            if self._cb_failures >= 3:
                                self._cb_open_until = time.time() + 20
                                self._logger.warning(f"{req_id} opening circuit due to consecutive backend errors")
                            msg = f"❌ Backend error {r.status_code}: {r.text[:500]}"
                            yield msg
                            http_code_local = r.status_code
                            error_msg = msg
                            return
                        current = ""
                        last_event = None
                        for raw in r.iter_lines(decode_unicode=True):
                            if self._cancel_requested:
                                yield "(stopped)"
                                break
                            if not raw:
                                continue
                            if raw.startswith("event: "):
                                last_event = raw[len("event: "):].strip()
                                continue
                            if raw.startswith("data: "):
                                try:
                                    data = json.loads(raw[len("data: "):])
                                except Exception:
                                    continue
                                if last_event == "response.output_text.delta":
                                    delta = data.get("delta", "")
                                    if isinstance(delta, str) and delta:
                                        current += delta
                                        yield current
                                elif last_event == "response.completed":
                                    break
                        if not current:
                            yield "(no streamed text)"
                        else:
                            # Success: reset failures
                            self._cb_failures = 0
                        http_code_local = 200 if r.ok else r.status_code
                        out_text_local = current
                        success_flag = bool(r.ok)
                        self._current_stream_resp = None
                        return
                    else:
                        r = sess.post(url, json=payload, timeout=120)
                        if not r.ok:
                            self._cb_failures += 1
                            if self._cb_failures >= 3:
                                self._cb_open_until = time.time() + 20
                                self._logger.warning(f"{req_id} opening circuit due to consecutive backend errors")
                            msg = f"❌ Backend error {r.status_code}: {r.text[:500]}"
                            error_msg = msg
                            http_code_local = r.status_code
                            return msg
                        data = r.json()
                        texts = []
                        for item in data.get("output", []):
                            if item.get("type") == "message" and item.get("role") == "assistant":
                                content = item.get("content", [])
                                for c in content:
                                    if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                                        texts.append(c["text"])
                        if not texts:
                            meta = data.get("metadata", {})
                            return meta.get("__debug_output", "(no text output from backend)")
                        # Success: reset failures
                        self._cb_failures = 0
                        http_code_local = r.status_code
                        out_text_local = "\n".join(texts).strip()
                        success_flag = True
                        return out_text_local
                except Exception as e:
                    self._cb_failures += 1
                    if self._cb_failures >= 3:
                        self._cb_open_until = time.time() + 20
                        self._logger.warning(f"{req_id} opening circuit due to exceptions contacting backend")
                    error_msg = f"❌ Error calling backend: {e}"
                    return error_msg

            # Local mode
            if self.model is None:
                return "❌ Model not loaded. Please load the model first."
            
            if not message.strip():
                return "Please enter a message."
            
            try:
                # Use the exact same approach that worked in our tests
                messages = [{"role": "user", "content": message}]

                # Apply chat template (uses harmony format automatically)
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )

                # Move to the right device and prepare attention mask
                input_ids = inputs.to(self.model.device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.model.device)

                # Determine model dtype for autocast
                try:
                    model_dtype = next(self.model.parameters()).dtype
                except StopIteration:
                    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                # Generate with autocast to avoid dtype mismatches
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.autocast(device_type="cuda", dtype=model_dtype):
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                pad_token_id=self.tokenizer.eos_token_id,
                                use_cache=True
                            )
                    else:
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )

                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs.shape[1]:], 
                    skip_special_tokens=True
                )

                # Clean up
                del outputs
                self.clear_memory()

                out_text_local = response.strip()
                success_flag = True
                return out_text_local

            except Exception as e:
                self.clear_memory()
                error_msg = f"❌ Error generating response: {str(e)}"
                return error_msg
        finally:
            # Release inflight guard and log duration
            self._inflight = max(0, self._inflight - 1)
            dt = time.time() - t_start
            self._logger.info(f"{req_id} done in {dt:.2f}s inflight={self._inflight}")
            # metrics
            try:
                latency_ms = int(dt * 1000)
                self._record_metrics(success_flag, latency_ms, http_code_local, out_text_local, error_msg)
                # DB + backend telemetry for requests
                try:
                    payload_len = len(message or "")
                    response_len = len(out_text_local or "") if isinstance(out_text_local, str) else 0
                    self.db_log_request("/v1/responses" if self.use_backend else "local.generate", http_code_local, latency_ms, payload_len, response_len)
                    self.send_telemetry("debug", "request", {"route": "/v1/responses" if self.use_backend else "local.generate", "http": http_code_local, "latency_ms": latency_ms})
                except Exception:
                    pass
            except Exception:
                pass

    def test_backend(self, url: str) -> str:
        """Check backend connectivity and basic /v1/responses shape without heavy load."""
        if requests is None:
            return "❌ Python 'requests' package not installed. Install 'requests' or switch to Local mode."
        url = self._normalize_backend_url(url)
        try:
            root = requests.get(url + "/", timeout=8)
            if not root.ok:
                return f"⚠️ GET / -> HTTP {root.status_code}"
        except Exception as e:
            return f"❌ GET / failed: {e}"
        try:
            ping = requests.post(
                url + "/v1/responses",
                json={"input": "ping", "stream": False, "max_output_tokens": 8, "metadata": {"__debug": True}},
                timeout=12,
            )
            if not ping.ok:
                return f"⚠️ POST /v1/responses -> HTTP {ping.status_code}: {ping.text[:160]}"
            data = ping.json()
            # Minimal shape check
            if "output" in data:
                return "✅ Backend healthy (GET / ok, POST /v1/responses ok)"
            return "⚠️ Backend responded but output shape unexpected"
        except Exception as e:
            return f"❌ POST /v1/responses failed: {e}"

# Initialize interface
gpt_interface = FixedGPTInterface()

def create_fixed_interface():
    """Create the fixed interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="GPT-OSS 20B Fixed", analytics_enabled=False) as demo:
        # Provide a local manifest.json to avoid dev-tunnel CORS/redirects
        try:
            import fastapi
            from fastapi.responses import JSONResponse
            from starlette.middleware.cors import CORSMiddleware
            app = demo.app if hasattr(demo, 'app') else None
            if app and isinstance(app, fastapi.FastAPI):
                # Allow all origins (local dev convenience)
                try:
                    app.add_middleware(
                        CORSMiddleware,
                        allow_origins=["*"],
                        allow_credentials=False,
                        allow_methods=["*"],
                        allow_headers=["*"],
                    )
                except Exception:
                    pass
                # Serve a local manifest explicitly to avoid dev tunnel redirects
                @app.get('/manifest.json')
                def _local_manifest():
                    return JSONResponse({
                        "name": "GPT-OSS 20B Fixed",
                        "short_name": "GPT-OSS",
                        "display": "standalone",
                        "start_url": "/",
                        "icons": []
                    })
                # Provide a tiny favicon to keep some UAs from probing cross-origin
                @app.get('/favicon.ico')
                def _favicon():
                    return JSONResponse({}, media_type="image/x-icon")
        except Exception:
            pass

        gr.Markdown("""
        # 🛠️ GPT-OSS 20B Fixed Interface
        
    This uses the exact same loading approach that was working in our memory-optimized tests.

    Tip: if using a VS Code dev tunnel, some browsers block the app manifest by CORS and the page may appear to "load forever". Access via http://127.0.0.1:7862 or set the environment variable GRADIO_SHARE=1 to get a public gradio.live link that bypasses the tunnel.
        """)
        # Patch manifest on the client to avoid dev tunnel redirect/auth CORS by removing manifest entirely
        gr.HTML(
            """
            <script>
            (function(){
              try {
                function removeAllManifests(){
                  Array.from(document.querySelectorAll('link[rel="manifest"]'))
                    .forEach(l=>{ try{ l.parentElement && l.parentElement.removeChild(l);}catch(e){} });
                }
                if (document.readyState === 'loading'){
                  document.addEventListener('DOMContentLoaded', removeAllManifests);
                } else {
                  removeAllManifests();
                }
              } catch(e){}
            })();
            </script>
            """,
            visible=True
        )
        
        with gr.Row():
            load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
            status_box = gr.Textbox(
                value="⭕ Model not loaded", 
                label="Status", 
                interactive=False,
                scale=2
            )
            refresh_btn = gr.Button("🔄 Refresh Info")
            test_btn = gr.Button("🧪 Test Backend")
            health_btn = gr.Button("📈 Status Overview")
            logs_btn = gr.Button("🧾 Show Recent Logs")
            merged_logs_btn = gr.Button("🧾 Show All Logs")
            monitor_btn = gr.Button("🧭 Monitor Snapshot")
            stop_btn = gr.Button("⛔ Stop Generation", variant="stop")
        
        with gr.Row():
            with gr.Column():
                message_box = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=3
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")
                
                response_box = gr.Textbox(
                    label="GPT-OSS Response",
                    lines=8,
                    interactive=False
                )
                logs_box = gr.Textbox(
                    label="Recent Logs",
                    lines=8,
                    interactive=False
                )
                events_terminal = gr.Textbox(
                    label="Events Terminal (merged)",
                    lines=18,
                    interactive=False
                )
                sys_usage = gr.Textbox(
                    label="System Usage",
                    lines=6,
                    interactive=False
                )
                gr.Markdown("_Tip: install optional packages for richer stats: `pip install psutil nvidia-ml-py`_")
                monitor_box = gr.Textbox(
                    label="Monitor Snapshot",
                    lines=12,
                    interactive=False
                )
                disable_timers_toggle = gr.Checkbox(label="Disable auto-refresh timers (dev tunnels)", value=False)
                telemetry_table = gr.Textbox(
                    label="Telemetry DB (events/errors/requests)",
                    lines=10,
                    interactive=False
                )
                telemetry_select = gr.Radio(["events","errors","requests"], value="events", label="Telemetry Table")
                telemetry_refresh = gr.Button("🔁 Refresh Telemetry")
            
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                mode = gr.Radio(["Backend API", "Local Transformers"], value="Backend API", label="Mode")
                backend_url = gr.Textbox(label="Backend URL", value="http://127.0.0.1:8000", placeholder="http://host:port")
                
                max_tokens = gr.Slider(
                    8, 128, 64, 
                    label="Max Tokens",
                    info="Keep low to save memory"
                )
                
                temperature = gr.Slider(
                    0.1, 2.0, 1.0, 
                    label="Temperature"
                )
                
                top_p = gr.Slider(
                    0.1, 1.0, 0.9, 
                    label="Top P"
                )
                stream = gr.Checkbox(label="Stream output", value=False)

                gpu_mem_cap = gr.Slider(
                    8, 20, 12,
                    label="GPU VRAM cap (GiB)",
                    info="How much VRAM to allow for the model; lower if OOM, higher if plenty free"
                )
                gpu_only = gr.Checkbox(
                    label="GPU-only (strict, no offload)",
                    value=True
                )
                
                gr.Markdown("""
                ### Instructions
                1. Click **Load Model** (takes 2-5 min)
                2. Wait for "Model loaded successfully"
                3. Type your message and click **Send**
                
                **Memory Tips:**
                - Close other GPU apps first
                - Use shorter responses (lower max tokens)
                - This uses the exact approach that worked before
                """)
        
        # Event handlers
        def handle_load(selected_mode, url, gpu_mem_gib, gpu_only_flag):
            # Update interface state
            gpt_interface.use_backend = (selected_mode == "Backend API")
            gpt_interface.backend_url = url.strip() or gpt_interface.backend_url
            gen = gpt_interface.load_model(int(gpu_mem_gib), bool(gpu_only_flag))
            for status_update in gpt_interface.run_guarded_event("load_model", gen, expected_max_seconds=600.0):
                yield status_update
        
        def handle_chat(selected_mode, url, message, max_tokens, temperature, top_p, stream_flag):
            gpt_interface.use_backend = (selected_mode == "Backend API")
            gpt_interface.backend_url = url.strip() or gpt_interface.backend_url
            if not message.strip():
                return "Please enter a message."
            if gpt_interface.loading and not gpt_interface.use_backend:
                return "⏳ Model is still loading... please wait until the status shows 'Model loaded successfully'."
            # Stream by yielding directly when enabled so Gradio doesn't show a generator object
            if stream_flag:
                gen = gpt_interface.chat(message, max_tokens, temperature, top_p, True)
                for chunk in gpt_interface.run_guarded_event("chat", gen, expected_max_seconds=180.0):
                    yield chunk
                return
            else:
                # Non-stream path: call synchronously to avoid queue/long-poll in tunnels
                return gpt_interface.chat(message, max_tokens, temperature, top_p, False)
        
        def clear_chat():
            return "", ""
        
        # Wire up events
        # Disable buttons while loading to reduce breakage
        def _disable_controls():
            # Disable all primary controls
            return (
                gr.update(interactive=False), # load_btn
                gr.update(interactive=False), # send_btn
                gr.update(interactive=False), # refresh_btn
                gr.update(interactive=False), # test_btn
                gr.update(interactive=False), # health_btn
                gr.update(interactive=False), # logs_btn
                gr.update(interactive=False), # merged_logs_btn
                gr.update(interactive=False), # monitor_btn
                gr.update(interactive=False), # stop_btn
            )
        def _enable_controls():
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        load_btn.click(
            _disable_controls,
            outputs=[load_btn, send_btn, refresh_btn, test_btn, health_btn, logs_btn, merged_logs_btn, monitor_btn, stop_btn],
        ).then(
            handle_load,
            inputs=[mode, backend_url, gpu_mem_cap, gpu_only],
            outputs=[status_box]
        ).then(
            _enable_controls,
            outputs=[load_btn, send_btn, refresh_btn, test_btn, health_btn, logs_btn, merged_logs_btn, monitor_btn, stop_btn],
        )

        refresh_btn.click(
            lambda: gpt_interface.device_summary(),
            outputs=[status_box]
        )

        test_btn.click(
            lambda m, url: gpt_interface.test_backend(url) if m == "Backend API" else "ℹ️ Switch mode to Backend API to test.",
            inputs=[mode, backend_url],
            outputs=[status_box]
        )
        health_btn.click(
            lambda: gpt_interface.health_summary(),
            outputs=[status_box]
        )
        logs_btn.click(
            lambda: gpt_interface.recent_logs(),
            outputs=[logs_box]
        )
        merged_logs_btn.click(
            lambda: gpt_interface.recent_logs_merged(True, 200),
            outputs=[events_terminal]
        )
        monitor_btn.click(
            lambda: gpt_interface.monitor_snapshot(),
            outputs=[monitor_box]
        )
        stop_btn.click(
            lambda: gpt_interface.stop_generation(),
            outputs=[status_box]
        )

        # Optional: disable timers in dev tunnels to avoid /queue 504s (env or UI)
        disable_timers = os.environ.get("DISABLE_TIMERS", "false").lower() in ("1", "true", "yes")
        # let UI toggle override env if changed later via reload
        if disable_timers_toggle.value:
            disable_timers = True
        if not disable_timers:
            # Live system usage refresh every 5s
            def _sys_usage():
                return gpt_interface.system_usage_summary()
            sys_timer = gr.Timer(5.0)
            sys_timer.tick(_sys_usage, outputs=[sys_usage])

            # Live events terminal refresh every 4s
            def _events():
                return gpt_interface.recent_logs_merged(True, 200)
            ev_timer = gr.Timer(4.0)
            ev_timer.tick(_events, outputs=[events_terminal])

            # Live monitor refresh every 6s
            def _mon():
                return gpt_interface.monitor_snapshot()
            mon_timer = gr.Timer(6.0)
            mon_timer.tick(_mon, outputs=[monitor_box])
        else:
            gr.Markdown("Timers disabled (DISABLE_TIMERS=1). Use the buttons to refresh logs and metrics.")
        
        send_btn.click(
            _disable_controls,
            outputs=[load_btn, send_btn, refresh_btn, test_btn, health_btn, logs_btn, merged_logs_btn, monitor_btn, stop_btn],
        ).then(
            handle_chat,
            inputs=[mode, backend_url, message_box, max_tokens, temperature, top_p, stream],
            outputs=[response_box]
        ).then(
            _enable_controls,
            outputs=[load_btn, send_btn, refresh_btn, test_btn, health_btn, logs_btn, merged_logs_btn, monitor_btn, stop_btn],
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[message_box, response_box]
        )
        
        # Enter key support
        message_box.submit(
            handle_chat,
            inputs=[mode, backend_url, message_box, max_tokens, temperature, top_p, stream],
            outputs=[response_box]
        )

        # Telemetry viewer
        telemetry_refresh.click(
            lambda table: gpt_interface.db_recent(table, 50),
            inputs=[telemetry_select],
            outputs=[telemetry_table]
        )
    
    return demo

if __name__ == "__main__":
    print("🛠️ Starting GPT-OSS Fixed Interface...")
    print("Using the same memory-optimized approach that was working before")
    print("Make sure you have ~20GB free GPU memory!")
    
    demo = create_fixed_interface()
    # Allow overriding port via env, else use 7862 but fall back to auto if busy
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    env_share = os.environ.get("GRADIO_SHARE", "false").lower() in ("1", "true", "yes")
    env_host = os.environ.get("GRADIO_SERVER_NAME") or os.environ.get("HOST") or "0.0.0.0"
    server_port = int(env_port) if env_port else 7862
    try:
        demo.launch(
            server_name=env_host,
            server_port=server_port,
            share=env_share,
            inbrowser=False,
            show_error=True,
        )
    except OSError:
        # Port busy; retry on a random free port
        print(f"Port {server_port} is busy; retrying on a random free port...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=None,  # auto-pick
            share=False,
            inbrowser=False,
            show_error=True,
        )