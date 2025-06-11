from dataclasses import dataclass

import numpy as np

from iluvattnshun.prompter import PromptConfig, Prompter


@dataclass
class VariableRenamingConfig(PromptConfig):
    """Configuration for variable renaming prompts.

    Parameters:
        num_chains: Number of independent renaming chains
        depth: Maximum variable renaming chain length
    """

    num_chains: int
    depth: int


class VariableRenamingPrompter(Prompter[VariableRenamingConfig]):
    """Prompter for generating variable renaming exercises.

    Generates prompts where variables are renamed in chains, and the task
    is to evaluate the final variable in terms of the initial value.
    """

    def get_prompt(self) -> tuple[str, str]:
        """Samples a variable renaming prompt and answers.

        Key idea is that when redefining, only sample from variables which are
        not currently the most recent variable in any chain.

        A key limitation is that we don't allow for more than 25 chains.
        """
        assert self.config.num_chains <= 25, "We don't support more than 25 chains."

        chains: list[list[int | str]] = [[np.random.randint(0, 10)] for _ in range(self.config.num_chains)]
        prompt = ""

        while True:
            unfilled_chains = [i for i in range(self.config.num_chains) if len(chains[i]) < self.config.depth + 1]
            if len(unfilled_chains) == 0:
                break

            sampled_chain = np.random.choice(unfilled_chains)
            most_recent_vars = [chain[-1] for chain in chains]
            old_var = chains[sampled_chain][-1]
            new_var = np.random.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c not in most_recent_vars])
            chains[sampled_chain].append(new_var)

            prompt += f"{new_var}={old_var};"

        final_var_evals = [(str(chain[-1]), str(chain[0])) for chain in chains]
        var_to_eval = np.random.choice(len(final_var_evals))
        prompt += final_var_evals[var_to_eval][0] + "?"
        answer = final_var_evals[var_to_eval][1]

        return prompt, answer

    def tokenize(self, text: str) -> list[int]:
        """Tokenize the input text.

        Maps:
        - Numbers 0-9 -> tokens 0-9
        - Letters a-z -> tokens 10-35
        - '=' -> token 36
        - '?' -> token 37
        - ';' -> token 38
        """
        tokens = []
        for c in text:
            if c.isdigit():
                tokens.append(int(c))
            elif c.isalpha() and c.islower():
                tokens.append(ord(c) - ord("a") + 10)
            elif c == "=":
                tokens.append(36)
            elif c == "?":
                tokens.append(37)
            elif c == ";":
                tokens.append(38)
            else:
                raise ValueError(f"Unexpected character: {c}")
        return tokens


if __name__ == "__main__":
    config = VariableRenamingConfig(num_prompts=100, num_chains=5, depth=5, seed=42)
    prompter = VariableRenamingPrompter(config)
    prompter.make_dataset("data/var_rename")
