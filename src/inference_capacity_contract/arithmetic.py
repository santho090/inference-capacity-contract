"""Shared integer arithmetic for capacity generation and validation."""

from __future__ import annotations

from decimal import Decimal


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def usable_memory_bytes(memory_bytes: int, utilization: float) -> int:
    return int(Decimal(memory_bytes) * Decimal(str(utilization)))


def sequence_capacity(
    capacity_blocks: int,
    block_size_tokens: int,
    context_tokens: int,
    max_num_seqs: int | None,
) -> tuple[int, int]:
    blocks_per_sequence = ceil_div(context_tokens, block_size_tokens)
    max_sequences = capacity_blocks // blocks_per_sequence
    if max_num_seqs is not None:
        max_sequences = min(max_sequences, max_num_seqs)
    return blocks_per_sequence, max_sequences
