"""Own one native idle miner in the CUDA context of a tensor-parallel scheduler."""
import atexit
import ctypes
import json
import os
from pathlib import Path
import stat


class NativeIdle:
    def __init__(self, rank, config_path):
        import torch
        from sglang.srt.runtime_context import get_device

        with os.fdopen(os.open(config_path, os.O_RDONLY | os.O_NOFOLLOW)) as source:
            metadata = os.fstat(source.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
                    or metadata.st_mode & 0o077):
                raise ValueError("Native miner config must be an owner-only regular file")
            config = json.load(source)
        self.library = ctypes.CDLL(config.pop("library"))
        config["worker"] = f"{config.pop('worker_prefix')}{rank}"
        self.log_path = Path(config.pop("log_dir")) / f"native-{rank}-{os.getpid()}.jsonl"
        config["log"] = str(self.log_path)
        self.library.pearl_idle_start.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.library.pearl_idle_start.restype = ctypes.c_void_p
        self.library.pearl_idle_stop.argtypes = [ctypes.c_void_p]
        self.library.pearl_idle_stop.restype = None
        device = get_device().gpu_id
        with torch.cuda.device(device):
            torch.cuda.current_stream(device)
            error = ctypes.create_string_buffer(512)
            self.session = self.library.pearl_idle_start(
                device, json.dumps(config).encode(), error, len(error)
            )
        if not self.session:
            raise RuntimeError(f"Native Pearl startup failed: {error.value.decode(errors='replace')}")
        atexit.register(self.close)

    def close(self):
        if self.session:
            self.library.pearl_idle_stop(self.session)
            self.session = None
