"""Base logger class."""

import shutil
from time import time
from typing import Literal

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)


class Logger:
    """Creates beautiful colored console log frames."""

    def __init__(self, precision: int = 4, log_every_n_seconds: float = 30.0):
        self.precision = precision
        self.log_every_n_seconds = log_every_n_seconds
        self.start_time = time()
        self.step = 0
        self.last_log_time: dict[Literal["train", "val"], float] = {"train": 0.0, "val": 0.0}

    def _format_number(self, value: float | int) -> str:
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.{self.precision}f}"

    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def _get_terminal_size(self) -> tuple[int, int]:
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24

    def log(self, metrics: dict[str, float], mode: Literal["train", "val"]) -> None:
        """Log metrics to the console.

        Args:
            metrics: Format {name: float value}
            mode: Mode to log (currently only train and val are supported)
        """
        curr_time = time()
        if curr_time - self.last_log_time[mode] < self.log_every_n_seconds:
            return
        iter_time = curr_time - self.last_log_time[mode]
        self.last_log_time[mode] = curr_time

        self.step += 1
        mode_color = Fore.GREEN if mode == "train" else Fore.YELLOW
        mode_str = f"{mode_color}{mode.upper():<5}{Style.RESET_ALL}"
        elapsed_str = f"{Fore.CYAN}{self._format_time(time() - self.start_time)}{Style.RESET_ALL}"
        step_str = f"{Fore.MAGENTA}Step {self.step:,}{Style.RESET_ALL}"
        iter_str = f"{Fore.MAGENTA}Iter {iter_time:.2f}s{Style.RESET_ALL}"

        term_width, term_height = self._get_terminal_size()

        header = f"{mode_str} | {step_str} | {iter_str} | Time: {elapsed_str}"
        horizontal_rule = f"{Fore.BLUE}{'─' * term_width}{Style.RESET_ALL}"

        metric_lines = [
            f"{Fore.WHITE}{name:<20}:{Style.RESET_ALL} {mode_color}{self._format_number(value)}{Style.RESET_ALL}"
            for name, value in metrics.items()
        ]

        frame = [horizontal_rule, header.center(term_width), horizontal_rule]
        frame.extend(metric_lines)

        while len(frame) < term_height - 1:
            frame.append("")

        frame.append(f"{Fore.BLUE}{'╶' * term_width}{Style.RESET_ALL}")
        print("\n".join(frame))
