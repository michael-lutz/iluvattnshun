"""Base logger class."""

import shutil
from time import time
from typing import Literal

from colorama import Fore, Style
from colorama import init as colorama_init
from torch.utils.tensorboard.writer import SummaryWriter

colorama_init(autoreset=True)


class Logger:
    """Creates beautiful colored console log frames and logs to TensorBoard."""

    def __init__(
        self,
        precision: int = 4,
        log_every_n_seconds: float = 30.0,
        tensorboard_logdir: str | None = None,
    ):
        self.precision = precision
        self.log_every_n_seconds = log_every_n_seconds
        self.start_time = time()
        self.step: dict[Literal["train", "val"], int] = {"train": 0, "val": 0}

        # not all logs result in a new message, but want to track iter time
        self.last_log_time: dict[Literal["train", "val"], float] = {
            "train": self.start_time,
            "val": self.start_time,
        }
        self.last_call_time: dict[Literal["train", "val"], float] = {
            "train": self.start_time,
            "val": self.start_time,
        }

        self.tb_writer = SummaryWriter(tensorboard_logdir) if tensorboard_logdir is not None else None

    def format_number(self, value: float | int) -> str:
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.{self.precision}f}"

    def format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def get_terminal_size(self) -> tuple[int, int]:
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24

    def log_to_tensorboard(
        self,
        metrics: dict[str, float | str],
        mode: Literal["train", "val"],
        header: dict[str, float | str],
    ) -> None:
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, float):
                    self.tb_writer.add_scalar(f"{mode}/{k}", v, self.step[mode])

            for k, v in header.items():
                if isinstance(v, float):
                    self.tb_writer.add_scalar(f"{mode}/{k}", v, self.step[mode])

    def log_to_console(
        self,
        metrics: dict[str, float | str],
        mode: Literal["train", "val"],
        header: dict[str, float | str],
    ) -> None:
        term_width, term_height = self.get_terminal_size()
        mode_color = Fore.GREEN if mode == "train" else Fore.YELLOW
        mode_str = f"{mode_color}{mode.upper():<5}{Style.RESET_ALL}"
        header_str = f"{mode_str} | " + " | ".join([f"{k}: {Fore.CYAN}{v}{Style.RESET_ALL}" for k, v in header.items()])
        horizontal_rule = f"{Fore.BLUE}{'─' * term_width}{Style.RESET_ALL}"

        metric_lines = []
        for name, value in metrics.items():
            if isinstance(value, float):
                metric_lines.append(
                    f"{Fore.WHITE}{name:<20}:{Style.RESET_ALL} {mode_color}{self.format_number(value)}{Style.RESET_ALL}"
                )
            else:
                metric_lines.append(f"{Fore.WHITE}{name:<20}:{Style.RESET_ALL} {mode_color}{value}{Style.RESET_ALL}")

        frame = ["\n", horizontal_rule, header_str, horizontal_rule]
        frame.extend(metric_lines)

        while len(frame) < term_height - 1:
            frame.append("")

        frame.append(f"{Fore.BLUE}{'╶' * term_width}{Style.RESET_ALL}")
        print("\n".join(frame))

    def log(
        self,
        metrics: dict[str, float | str],
        mode: Literal["train", "val"],
        header: dict[str, float | str] | None = None,
    ) -> None:
        curr_time = time()
        iter_time = curr_time - self.last_call_time[mode]
        self.step[mode] += 1
        self.last_call_time[mode] = curr_time

        if curr_time - self.last_log_time[mode] < self.log_every_n_seconds:
            return

        self.last_log_time[mode] = curr_time

        if header is None:
            header = {}

        header["step"] = self.step[mode]
        header["time"] = self.format_time(curr_time - self.start_time)
        header["iter_time"] = self.format_number(iter_time)

        self.log_to_tensorboard(metrics, mode, header)
        self.log_to_console(metrics, mode, header)

    def close(self) -> None:
        """Clean up resources like TensorBoard writers."""
        if self.tb_writer is not None:
            self.tb_writer.close()
