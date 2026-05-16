from __future__ import annotations

from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    """Tracks token usage per agent across interactions."""

    def __init__(self) -> None:
        self._usage: dict[str, list[TokenUsage]] = {}

    def record(
        self,
        agent: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        if agent not in self._usage:
            self._usage[agent] = []
        self._usage[agent].append(
            TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
            )
        )

    def get_total(self, agent: str) -> TokenUsage:
        records = self._usage.get(agent, [])
        if not records:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=sum(r.prompt_tokens for r in records),
            completion_tokens=sum(r.completion_tokens for r in records),
            model=records[-1].model,
        )

    def get_grand_total(self) -> TokenUsage:
        all_records: list[TokenUsage] = []
        for records in self._usage.values():
            all_records.extend(records)
        if not all_records:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=sum(r.prompt_tokens for r in all_records),
            completion_tokens=sum(r.completion_tokens for r in all_records),
            model="",
        )

    def is_within_budget(self, agent: str, budget: int = 4600) -> bool:
        total = self.get_total(agent)
        return total.total_tokens <= budget

    def summary(self) -> dict[str, dict]:
        result = {}
        for agent in self._usage:
            total = self.get_total(agent)
            result[agent] = {
                "prompt": total.prompt_tokens,
                "completion": total.completion_tokens,
                "total": total.total_tokens,
            }
        return result
