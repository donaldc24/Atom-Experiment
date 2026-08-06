"""Model components for independent LoRA and shared-atom experiments."""

from .atoms import (
    AtomLinear,
    TaskContext,
    atom_coefficient_l1,
    atom_parameter_count,
    atom_regularization_loss,
    clear_atom_top_k,
    coefficient_l1_penalty,
    coefficient_l1_regularization,
    iter_atom_layers,
    set_atom_top_k,
)
from .classifier import BertTaskClassifier
from .injection import (
    CANONICAL_TARGET_SUFFIXES,
    QUERY_SUFFIX,
    VALUE_SUFFIX,
    adapter_state_dict,
    canonical_target_suffixes,
    extract_adapter_state_dict,
    inject_atoms,
    inject_lora,
    load_adapter_state_dict,
    resolve_target_linears,
)
from .lora import LoRALinear, iter_lora_layers, lora_parameter_count

__all__ = [
    "AtomLinear",
    "BertTaskClassifier",
    "CANONICAL_TARGET_SUFFIXES",
    "LoRALinear",
    "QUERY_SUFFIX",
    "TaskContext",
    "VALUE_SUFFIX",
    "adapter_state_dict",
    "atom_coefficient_l1",
    "atom_parameter_count",
    "atom_regularization_loss",
    "canonical_target_suffixes",
    "clear_atom_top_k",
    "coefficient_l1_penalty",
    "coefficient_l1_regularization",
    "extract_adapter_state_dict",
    "inject_atoms",
    "inject_lora",
    "iter_atom_layers",
    "iter_lora_layers",
    "load_adapter_state_dict",
    "lora_parameter_count",
    "resolve_target_linears",
    "set_atom_top_k",
]
