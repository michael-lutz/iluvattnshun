"""DAG datagen"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from iluvattnshun.prompter import PromptConfig, Prompter
from iluvattnshun.trainer import TrainerConfig
from iluvattnshun.types import TensorTree


@dataclass
class DAGConfig(PromptConfig, TrainerConfig):
    """Configuration for variable renaming prompts."""

    # model
    num_layers: int
    """Number of transformer layers."""
    dim_model: int
    """Dimension of the model."""
    num_heads: int
    """Number of attention heads."""

    # data generation
    num_nodes: int
    """Total number of nodes in the DAG."""
    min_ranks: int
    """Minimum number of ranks (layers) in the DAG."""
    max_ranks: int
    """Maximum number of ranks (layers) in the DAG."""
    min_per_rank: int
    """Minimum number of nodes per rank."""
    max_per_rank: int
    """Maximum number of nodes per rank."""
    num_errors: int
    """Number of error nodes to add."""
    min_outdeg: int
    """Minimum out-degree for each node."""
    max_outdeg: int
    """Maximum out-degree for each node."""
    train_size: int
    """Number of training examples."""
    test_size: int
    """Number of test examples."""
    dataset_path: str
    """Path to the dataset."""

    def data_hash_params(self) -> list[str]:
        return ["num_chains", "chain_length", "train_size", "test_size"]


class DAGPrompter(Prompter[DAGConfig]):
    """Prompter for generating variable renaming exercises.

    Generates prompts where variables are renamed in chains, and the task
    is to evaluate the final variable in terms of the initial value.
    """

    def get_prompt(self, rng: np.random.Generator) -> str:
        """Generate one DAG prompt."""

        cfg = self.config

        assert cfg.num_errors <= 10, "We don't support more than 10 errors."
        assert cfg.max_ranks * cfg.min_per_rank <= cfg.num_nodes, "Impossible constraints"

        num_ranks: int = int(rng.integers(cfg.min_ranks, cfg.max_ranks + 1))

        rank_sizes: list[int] = [cfg.min_per_rank] * num_ranks
        remaining_nodes = cfg.num_nodes - sum(rank_sizes)
        while remaining_nodes > 0:
            idx: int = int(rng.integers(0, num_ranks))
            if rank_sizes[idx] < cfg.max_per_rank:
                rank_sizes[idx] += 1
                remaining_nodes -= 1

        all_nodes = list("abcdefghijklmnopqrstuvwxyz")[: cfg.num_nodes]
        rng.shuffle(all_nodes)

        ranks: list[list[str]] = []
        idx = 0
        for size in rank_sizes:
            ranks.append(all_nodes[idx : idx + size])
            idx += size

        error_nodes = list(range(cfg.num_errors))
        rng.shuffle(error_nodes)
        ranks.append(error_nodes)

        edges: set[tuple[str, str]] = set()
        parents: dict[str, set[str]] = defaultdict(set)
        root_ancestors: dict[str, set[str]] = defaultdict(set)

        def _add_edge(src: str, dst: str) -> None:
            """Insert directed edge `src -> dst` and update parent tables."""
            edges.add((src, dst))
            parents[dst].add(src)

        start_node = rng.choice(ranks[0])
        end_node = rng.choice(ranks[-1])

        previous = start_node
        for intermediate_nodes in ranks[1:-1]:
            nxt = rng.choice(intermediate_nodes)
            _add_edge(previous, nxt)
            previous = nxt
        _add_edge(previous, end_node)

        total_ranks = len(ranks)
        for layer_idx, from_nodes in enumerate(ranks[:-1]):
            to_nodes = ranks[layer_idx + 1]

            min_out, max_out = cfg.min_outdeg, min(cfg.max_outdeg, len(to_nodes))

            for src in from_nodes:
                for p in parents[src]:
                    root_ancestors[src].update(root_ancestors[p])
                    if layer_idx == 1:
                        root_ancestors[src].add(p)

                allowed_targets = to_nodes
                if layer_idx == total_ranks - 2:
                    allowed_targets = [t for t in to_nodes if not (root_ancestors[t] | root_ancestors[src])]

                degree = int(rng.integers(min_out, max_out + 1))
                chosen = rng.choice(allowed_targets, size=min(degree, len(allowed_targets)), replace=False)
                for dst in chosen:
                    _add_edge(src, dst)
                    if layer_idx == total_ranks - 2:
                        root_ancestors[dst].update(root_ancestors[src])

        edge_str = ";".join(f"{u}>{v}" for u, v in edges)
        prompt_text = f"{start_node}>{end_node}|" + edge_str

        return prompt_text

    @property
    def _tokenization_map(self) -> dict[str, int]:
        char_to_token: dict[str, int] = {}
        for i in range(10):
            char_to_token[str(i)] = i
        for c in "abcdefghijklmnopqrstuvwxyz":
            char_to_token[c] = ord(c) - ord("a") + 10
        char_to_token[">"] = 36
        char_to_token["|"] = 37
        char_to_token[";"] = 38
        char_to_token["."] = -1
        return char_to_token

    def tokenize(self, text: str) -> list[int]:
        """Tokenize the input text.

        Maps:
        - Numbers 0-9 -> tokens 0-9
        - Letters a-z -> tokens 10-35
        - '>' -> token 36
        - '|' -> token 37
        - ';' -> token 38
        - '.' -> token -1 (mask)
        """
        tokens = []
        for c in text:
            tokens.append(self._tokenization_map[c])
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        """Detokenize the input tokens.

        Maps:
        - Tokens 0-9 -> Numbers 0-9
        - Tokens 10-35 -> Letters a-z
        - Tokens 36-38 -> '>', '|', ';'
        - Token -1 -> '.' (mask)
        """
        inverse_tokenization_map: dict[int, str] = {v: k for k, v in self._tokenization_map.items()}
        return "".join([inverse_tokenization_map[token] for token in tokens])
