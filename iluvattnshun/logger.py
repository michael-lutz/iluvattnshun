"""Base logger class."""

import os
import shutil
from datetime import datetime
from time import time
from typing import Literal

from colorama import Fore, Style
from colorama import init as colorama_init
from torch.utils.tensorboard.writer import SummaryWriter

colorama_init(autoreset=True)

KeyedMetric = dict[str | int | bool, float | int | bool]
"""E.g. {'A': 10, 'B': 20, 'C': 30}."""
Loggable = float | str | bool | KeyedMetric
"""Loggable types are float, str, bool, or a dict of str, int, or bool to float, int, or bool."""


class Logger:
    """Creates beautiful colored console log frames and logs to TensorBoard."""

    def __init__(
        self,
        tensorboard_logdir: str,
        precision: int = 4,
        log_every_n_seconds: float = 30.0,
        log_at_start: bool = True,
        name: str | None = None,
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

        # creating run directory with next run number
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(tensorboard_logdir):
            os.makedirs(tensorboard_logdir, exist_ok=True)
        existing_runs = [
            d
            for d in os.listdir(tensorboard_logdir)
            if d.startswith("run_") and os.path.isdir(os.path.join(tensorboard_logdir, d))
        ]
        run_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
        name = timestamp if name is None else name
        self.run_name = f"run_{next_run_number}_{name}"
        self.log_dir: str = os.path.join(tensorboard_logdir, self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb_writer = SummaryWriter(self.log_dir)
        self.log_at_start = log_at_start

    def format_value(self, value: float | int | bool) -> str:
        """Format value for logging."""
        if isinstance(value, bool):
            return "T" if value else "F"
        elif isinstance(value, float):
            # use scientific notation for very small numbers
            if abs(value) < 10 ** (-self.precision + 1) and value != 0:
                return f"{value:.{self.precision}e}"
            else:
                return f"{value:.{self.precision}f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")

    def format_time(self, seconds: float) -> str:
        """Format time in hours, minutes, and seconds."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def get_terminal_size(self) -> tuple[int, int]:
        """Returns the terminal size."""
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24

    def write_metrics_to_tensorboard(
        self,
        metrics: dict[str, Loggable],
        mode: Literal["train", "val"],
        header: dict[str, str | int | bool],
    ) -> None:
        """Log metrics and header to TensorBoard.

        Args:
            metrics: Metrics to log
            mode: Mode to log to
            header: Header to log
        """
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, float) or isinstance(v, int) or isinstance(v, bool):
                    self.tb_writer.add_scalar(f"{mode}/{k}", v, self.step[mode])
                elif isinstance(v, str):
                    self.tb_writer.add_text(f"{mode}/{k}", v, self.step[mode])
                elif isinstance(v, dict):
                    self.tb_writer.add_scalars(f"{mode}/{k}", v, self.step[mode])

            for k, v in header.items():
                if isinstance(v, float) or isinstance(v, int):
                    self.tb_writer.add_scalar(f"{mode}/{k}", v, self.step[mode])
                elif isinstance(v, str):
                    self.tb_writer.add_text(f"{mode}/{k}", v, self.step[mode])

    def write_metrics_to_console(
        self,
        metrics: dict[str, Loggable],
        mode: Literal["train", "val"],
        header: dict[str, str | int | bool],
    ) -> None:
        """Log metrics and header to console.

        Args:
            metrics: Metrics to log
            mode: Mode to log to
            header: Header to log
        """
        term_width, term_height = self.get_terminal_size()
        mode_color = Fore.GREEN if mode == "train" else Fore.YELLOW
        mode_str = f"{mode_color}{mode.upper():<5}{Style.RESET_ALL}"
        header_str = f"{mode_str} | " + " | ".join([f"{k}: {Fore.CYAN}{v}{Style.RESET_ALL}" for k, v in header.items()])
        horizontal_rule = f"{Fore.BLUE}{'─' * term_width}{Style.RESET_ALL}"

        metric_lines = []
        for name, value in metrics.items():
            if isinstance(value, float) or isinstance(value, int):
                metric_lines.append(
                    f"{Fore.WHITE}{name:<20}:{Style.RESET_ALL} {mode_color}{self.format_value(value)}{Style.RESET_ALL}"
                )
            elif isinstance(value, str) or isinstance(value, bool):
                metric_lines.append(f"{Fore.WHITE}{name:<20}:{Style.RESET_ALL} {mode_color}{value}{Style.RESET_ALL}")
            elif isinstance(value, dict):
                metric_lines.append(f"{Fore.WHITE}{name:<20}:{Style.RESET_ALL}")
                for k, v in value.items():
                    metric_lines.append(
                        f"↳{Fore.WHITE}{k:<20}:{Style.RESET_ALL} {mode_color}{self.format_value(v)}{Style.RESET_ALL}"
                    )

        frame = ["\n", horizontal_rule, header_str, horizontal_rule]
        frame.extend(metric_lines)

        while len(frame) < term_height - 1:
            frame.append("")

        frame.append(f"{Fore.BLUE}{'╶' * term_width}{Style.RESET_ALL}")
        print("\n".join(frame))

    def log_metrics(
        self,
        metrics: dict[str, Loggable],
        mode: Literal["train", "val"],
        header: dict[str, str | int | bool] | None = None,
    ) -> None:
        """Log metrics and header to TensorBoard and console.

        Args:
            metrics: Metrics to log
            mode: Mode to log to
            header: Header to log
        """
        curr_time = time()
        iter_time = curr_time - self.last_call_time[mode]
        self.step[mode] += 1
        self.last_call_time[mode] = curr_time

        if curr_time - self.last_log_time[mode] < self.log_every_n_seconds and not (
            self.log_at_start and self.step[mode] == 1
        ):
            return

        self.last_log_time[mode] = curr_time

        if header is None:
            header = {}

        header["step"] = self.step[mode]
        header["time"] = self.format_time(curr_time - self.start_time)
        header["iter_time"] = self.format_value(iter_time)
        header["run_name"] = self.run_name

        self.write_metrics_to_tensorboard(metrics, mode, header)
        self.write_metrics_to_console(metrics, mode, header)

    def close(self) -> None:
        """Clean up resources like TensorBoard writers."""
        if self.tb_writer is not None:
            self.tb_writer.close()

    def write_text_to_console(self, name: str, text: str) -> None:
        """Write text to console."""
        term_width, term_height = self.get_terminal_size()
        horizontal_rule = f"{Fore.BLUE}{'─' * term_width}{Style.RESET_ALL}"
        header_str = f"{Fore.MAGENTA}{name}{Style.RESET_ALL}"
        horizontal_rule = f"{Fore.BLUE}{'─' * term_width}{Style.RESET_ALL}"
        text_lines = text.split("\n")

        frame = ["\n", horizontal_rule, header_str, horizontal_rule]
        frame.extend(text_lines)
        while len(frame) < term_height - 1:
            frame.append("")

        frame.append(f"{Fore.BLUE}{'╶' * term_width}{Style.RESET_ALL}")
        print("\n".join(frame))

    def log_text(self, name: str, text: str, save_to_file: bool = True, write_to_console: bool = False) -> None:
        """Log text to TensorBoard and optionally save to a file.

        Args:
            name: Name of the file (e.g. "script.py" or "config.txt")
            text: Text content to log
            save_to_file: Whether to save the text to a file in the log directory
        """
        assert self.tb_writer is not None
        self.tb_writer.add_text(name, f"```\n{text}\n```")
        if save_to_file and self.log_dir is not None:
            file_path = os.path.join(self.log_dir, f"{name}")
            with open(file_path, "w") as f:
                f.write(text)

        if write_to_console:
            self.write_text_to_console(name, text)
