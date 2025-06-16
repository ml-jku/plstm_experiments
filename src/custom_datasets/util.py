from typing import Any, Dict, TypeVar, Union, List, Tuple, Set
import copy
import dataclasses

T = TypeVar("T")


def apply_override(obj: T, override: Dict[str, Any], enforce: bool = False, exact_match: bool = True) -> T:
    """
    Recursively applies values from an override dictionary to an object or dictionary.

    This function can modify attributes of an object or keys in a dictionary based on the
    provided override dictionary. It handles nested structures by recursively applying
    overrides to nested objects or dictionaries.

    Parameters
    ----------
    obj : Any
        The target object or dictionary to be modified.
    override : Dict[str, Any]
        Dictionary containing the values to override in the target object.
    enforce : bool, default=False
        If True, will force setting attributes/keys even if they don't exist in the original object.
        If False, will only override existing attributes/keys.
    exact_match : bool, default=True
        If True and enforce is False, will raise ValueError when a key doesn't exist.
        If False, will silently ignore keys that don't exist.

    Returns
    -------
    T
        The modified object with overrides applied.

    Examples
    --------
    >>> class Config:
    ...     def __init__(self):
    ...         self.name = "default"
    ...         self.params = {"learning_rate": 0.01, "batch_size": 32}
    ...     def __repr__(self):
    ...         return f"Config(name='{self.name}', params={self.params})"
    >>> config = Config()
    >>> override = {"name": "custom", "params": {"learning_rate": 0.001}}
    >>> apply_override(config, override)
    Config(name='custom', params={'learning_rate': 0.001, 'batch_size': 32})

    >>> # Dictionary example
    >>> base_dict = {"model": {"type": "cnn", "layers": 3}, "training": {"epochs": 10}}
    >>> override_dict = {"model": {"layers": 5}, "training": {"batch_size": 64}}
    >>> apply_override(base_dict, override_dict, enforce=True)
    {'model': {'type': 'cnn', 'layers': 5}, 'training': {'epochs': 10, 'batch_size': 64}}

    >>> # Using enforce=True to add new attributes
    >>> config = Config()
    >>> override = {"new_param": "value", "params": {"optimizer": "adam"}}
    >>> apply_override(config, override, enforce=True)
    Config(name='default', params={'learning_rate': 0.01, 'batch_size': 32, 'optimizer': 'adam'})
    >>> config.new_param
    'value'

    >>> # Using exact_match=False to ignore non-existent keys
    >>> config = Config()
    >>> override = {"unknown": "value"}
    >>> apply_override(config, override, exact_match=False)
    Config(name='default', params={'learning_rate': 0.01, 'batch_size': 32})

    >>> # Using exact_match=True (default) raises ValueError for non-existent keys
    >>> config = Config()
    >>> override = {"unknown": "value"}
    >>> try:
    ...     apply_override(config, override)
    ... except ValueError:
    ...     print("ValueError raised as expected")
    ValueError raised as expected
    """
    if isinstance(override, dict):
        for sub, val in override.items():
            if hasattr(obj, sub):
                setattr(obj, sub, apply_override(getattr(obj, sub), val, enforce=enforce, exact_match=exact_match))
            else:
                try:
                    if sub in obj or enforce:
                        if sub not in obj:
                            obj[sub] = val
                        else:
                            obj[sub] = apply_override(obj[sub], val, enforce=enforce, exact_match=exact_match)
                except TypeError:
                    if enforce:
                        setattr(obj, sub, val)
                    elif exact_match:
                        raise ValueError
        return obj
    else:
        return override


def deepcopy_with_dataclasses(obj: Any) -> Any:
    """
    Performs a deep copy of an object while preserving dataclass types.

    Standard copy.deepcopy() can convert nested dataclasses to dictionaries.
    This function ensures that dataclass instances remain as dataclasses
    throughout the copying process.

    Parameters
    ----------
    obj : Any
        The object to be deep copied.

    Returns
    -------
    Any
        A deep copy of the input object with preserved dataclass types.

    Examples
    --------
    >>> from dataclasses import dataclass, field
    >>> @dataclass
    ... class NestedConfig:
    ...     value: int = 42
    ...
    >>> @dataclass
    ... class Config:
    ...     name: str = "default"
    ...     nested: NestedConfig = field(default_factory=NestedConfig)
    ...
    >>> original = Config()
    >>> copied = deepcopy_with_dataclasses(original)
    >>> copied.name = "modified"
    >>> copied.nested.value = 100
    >>> original.name, original.nested.value
    ('default', 42)
    >>> copied.name, copied.nested.value
    ('modified', 100)
    >>> isinstance(copied.nested, NestedConfig)
    True
    """
    # Handle None
    if obj is None:
        return None

    # Handle dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Create a new instance of the same dataclass type
        result = copy.copy(obj)
        # Deep copy all fields
        for field in dataclasses.fields(obj):
            field_name = field.name
            field_value = getattr(obj, field_name)
            setattr(result, field_name, deepcopy_with_dataclasses(field_value))
        return result

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: deepcopy_with_dataclasses(v) for k, v in obj.items()}

    # Handle lists
    elif isinstance(obj, list):
        return [deepcopy_with_dataclasses(item) for item in obj]

    # Handle tuples
    elif isinstance(obj, tuple):
        return tuple(deepcopy_with_dataclasses(item) for item in obj)

    # Handle sets
    elif isinstance(obj, set):
        return {deepcopy_with_dataclasses(item) for item in obj}

    # For other types, use standard deepcopy
    else:
        return copy.deepcopy(obj)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
