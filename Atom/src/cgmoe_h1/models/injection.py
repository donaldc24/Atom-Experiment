"""Safe, exact adapter injection and compact adapter checkpoint helpers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MethodType
from typing import Any

from torch import Tensor, nn

from .atoms import AtomLinear, TaskContext
from .lora import LoRALinear


# Internally every matcher includes the component boundary.  Short names from
# the YAML config are expanded to these values; they are never used as bare
# ``endswith("query")`` matches.
QUERY_SUFFIX = ".attention.self.query"
VALUE_SUFFIX = ".attention.self.value"
CANONICAL_TARGET_SUFFIXES = (QUERY_SUFFIX, VALUE_SUFFIX)
_SHORT_TARGETS = {"query": QUERY_SUFFIX, "value": VALUE_SUFFIX}


def canonical_target_suffixes(target_suffixes: Sequence[str]) -> tuple[str, ...]:
    """Normalize sanctioned target labels to exact attention-self suffixes."""

    if not target_suffixes:
        raise ValueError("target_suffixes must not be empty")
    normalized: list[str] = []
    for suffix in target_suffixes:
        if suffix in _SHORT_TARGETS:
            canonical = _SHORT_TARGETS[suffix]
        else:
            canonical = suffix if suffix.startswith(".") else f".{suffix}"
        if canonical not in CANONICAL_TARGET_SUFFIXES:
            raise ValueError(
                f"unsupported adapter target {suffix!r}; only full attention.self.query "
                "and attention.self.value suffixes are allowed"
            )
        normalized.append(canonical)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate target suffixes are not allowed: {target_suffixes!r}")
    return tuple(normalized)


def _matches_suffix(module_name: str, suffix: str) -> bool:
    # The equality case supports a small synthetic module rooted directly at
    # ``attention.self.query``.  The suffix case demands a literal dot boundary.
    return module_name == suffix[1:] or module_name.endswith(suffix)


def resolve_target_linears(
    model: nn.Module,
    target_suffixes: Sequence[str] = CANONICAL_TARGET_SUFFIXES,
) -> list[tuple[str, nn.Linear]]:
    """Resolve exact query/value targets and reject matching non-linear modules."""

    suffixes = canonical_target_suffixes(target_suffixes)
    matches: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not name or not any(_matches_suffix(name, suffix) for suffix in suffixes):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"adapter target {name!r} must be nn.Linear, got {type(module).__name__}"
            )
        matches.append((name, module))
    if not matches:
        readable = ", ".join(suffix[1:] for suffix in suffixes)
        raise ValueError(f"no linear modules matched exact target suffixes: {readable}")
    return matches


def _replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if child_name not in parent._modules:  # noqa: SLF001 - exact registered-child guard
        raise RuntimeError(f"cannot replace unregistered module path {path!r}")
    setattr(parent, child_name, replacement)


def _validate_expected_count(names: Sequence[str], expected_count: int | None) -> None:
    if expected_count is None:
        return
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer or None")
    if len(names) != expected_count:
        raise ValueError(f"expected {expected_count} adapter targets, resolved {len(names)}: {names}")


def inject_lora(
    model: nn.Module,
    target_suffixes: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    *,
    expected_count: int | None = None,
) -> list[str]:
    """Replace exact BERT attention query/value linears with LoRA wrappers."""

    targets = resolve_target_linears(model, target_suffixes)
    names = [name for name, _ in targets]
    _validate_expected_count(names, expected_count)
    replacements = [
        (name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        for name, base in targets
    ]
    for name, replacement in replacements:
        _replace_module(model, name, replacement)
    return names


def _task_context_for(model: nn.Module, task_ids: Sequence[str]) -> TaskContext:
    ids = tuple(task_ids)
    context = getattr(model, "task_context", None)
    if context is None:
        context = TaskContext(ids)
        setattr(model, "task_context", context)
    elif not isinstance(context, TaskContext):
        raise TypeError("model.task_context exists but is not a TaskContext")
    elif context.task_ids != ids:
        raise ValueError(
            f"model task context order {context.task_ids!r} does not match requested {ids!r}"
        )
    return context


def _set_active_task_on_model(
    model: nn.Module,
    task_id: str,
    top_k: int | None = None,
) -> None:
    model.task_context.set_active_task(task_id, top_k=top_k)


def _set_task_on_model(model: nn.Module, task_id: str, top_k: int | None = None) -> None:
    _set_active_task_on_model(model, task_id, top_k=top_k)


def _set_atom_top_k_on_model(model: nn.Module, top_k: int | None) -> None:
    model.task_context.set_top_k(top_k)
    for module in model.modules():
        if isinstance(module, AtomLinear):
            module.set_top_k(top_k)


def _clear_atom_top_k_on_model(model: nn.Module) -> None:
    model.task_context.clear_top_k()
    for module in model.modules():
        if isinstance(module, AtomLinear):
            module.clear_top_k()


def _install_context_api(model: nn.Module) -> None:
    """Give a bare encoder the same small task API as BertTaskClassifier."""

    if not hasattr(model, "set_active_task"):
        model.set_active_task = MethodType(_set_active_task_on_model, model)
    if not hasattr(model, "set_task"):
        model.set_task = MethodType(_set_task_on_model, model)
    if not hasattr(model, "set_atom_top_k"):
        model.set_atom_top_k = MethodType(_set_atom_top_k_on_model, model)
    if not hasattr(model, "clear_atom_top_k"):
        model.clear_atom_top_k = MethodType(_clear_atom_top_k_on_model, model)


def inject_atoms(
    model: nn.Module,
    task_ids: Sequence[str],
    target_suffixes: Sequence[str],
    atom_count: int,
    scaling: float = 1.0,
    *,
    expected_count: int | None = None,
) -> list[str]:
    """Replace exact query/value linears with one shared atom bank per layer."""

    targets = resolve_target_linears(model, target_suffixes)
    names = [name for name, _ in targets]
    _validate_expected_count(names, expected_count)
    context = _task_context_for(model, task_ids)
    replacements = [
        (
            name,
            AtomLinear(
                base,
                task_ids=task_ids,
                atom_count=atom_count,
                scaling=scaling,
                task_context=context,
            ),
        )
        for name, base in targets
    ]
    for name, replacement in replacements:
        _replace_module(model, name, replacement)
    _install_context_api(model)
    return names


def _adapter_state_keys(model: nn.Module, include_heads: bool) -> set[str]:
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        if isinstance(module, LoRALinear):
            keys.update({f"{prefix}lora_a.weight", f"{prefix}lora_b.weight"})
        elif isinstance(module, AtomLinear):
            keys.update(
                {
                    f"{prefix}atom_v",
                    f"{prefix}atom_u",
                    f"{prefix}coefficients",
                    f"{prefix}_extra_state",
                }
            )
    if include_heads:
        heads = getattr(model, "heads", None)
        if isinstance(heads, nn.Module):
            head_path = next(
                (name for name, module in model.named_modules() if module is heads),
                None,
            )
            if head_path is not None:
                prefix = f"{head_path}." if head_path else ""
                keys.update(f"{prefix}{name}" for name in heads.state_dict())
    return keys


def extract_adapter_state_dict(
    model: nn.Module,
    *,
    include_heads: bool = True,
) -> OrderedDict[str, Any]:
    """Extract adapters (and normally task heads) without frozen base weights."""

    full_state = model.state_dict()
    keys = _adapter_state_keys(model, include_heads)
    state: OrderedDict[str, Any] = OrderedDict()
    for key in full_state:
        if key not in keys:
            continue
        value = full_state[key]
        state[key] = value.detach().cpu().clone() if isinstance(value, Tensor) else deepcopy(value)
    return state


def load_adapter_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    include_heads: bool = True,
    strict: bool = True,
) -> None:
    """Load a compact state dict while preventing accidental base-weight writes."""

    expected = _adapter_state_keys(model, include_heads)
    supplied = set(state_dict)
    unexpected = sorted(supplied - expected)
    missing = sorted(expected - supplied)
    if unexpected:
        raise RuntimeError(f"non-adapter keys are not accepted: {unexpected}")
    if strict and missing:
        raise RuntimeError(f"adapter state dict is missing keys: {missing}")
    # strict=False here is intentional: the compact state omits every frozen
    # base tensor.  Our own key checks above retain strictness within its scope.
    model.load_state_dict(dict(state_dict), strict=False)


# Concise compatibility alias for checkpoint call sites.
adapter_state_dict = extract_adapter_state_dict


__all__ = [
    "CANONICAL_TARGET_SUFFIXES",
    "QUERY_SUFFIX",
    "VALUE_SUFFIX",
    "adapter_state_dict",
    "canonical_target_suffixes",
    "extract_adapter_state_dict",
    "inject_atoms",
    "inject_lora",
    "load_adapter_state_dict",
    "resolve_target_linears",
]
