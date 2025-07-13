from abc import abstractmethod
import argparse
from dataclasses import dataclass
import inspect
from typing import List, get_type_hints

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.datasets.dataset import Dataset
from nnt.trainer import TrainingArguments
from nnt.validators.validator import Validator
from typing import Type


class ExperimentConfig:
    name: str
    output_dir: str
    training_args: TrainingArguments
    description: str = ""


class Experiment:

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.prepare()

    @abstractmethod
    def prepare(self):
        """
        Prepare the experiment by loading the model, dataset, and any other necessary components.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def run(self):
        """
        Run the experiment using the provided configuration.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


def experiment_config_cli(config_class: Type, verbose: bool = True) -> ExperimentConfig:
    def get_all_attrs(cls_type, prefix=""):
        attrs = {}
        seen = set()

        for base in reversed(cls_type.__mro__):
            if hasattr(base, "__annotations__"):
                for attr, typ in base.__annotations__.items():
                    if attr not in seen:
                        seen.add(attr)
                        full_key = f"{prefix}{attr}"
                        default_value = getattr(base, attr, None)

                        # Check if it's a nested config-like class (i.e., has __annotations__)
                        if inspect.isclass(typ) and hasattr(typ, "__annotations__"):
                            # Recurse into sub-config
                            nested_attrs = get_all_attrs(typ, prefix=f"{full_key}.")
                            attrs.update(nested_attrs)
                        else:
                            attrs[full_key] = default_value
        return attrs

    def add_arguments(parser, attrs):
        for full_attr, default in attrs.items():
            arg_type = type(default) if default is not None and type(default) in [int, float, str, bool] else str
            parser.add_argument(
                f"--{full_attr}",
                type=arg_type,
                default=default,
                required=False,
            )

    all_attrs = get_all_attrs(config_class)
    parser = argparse.ArgumentParser(description=getattr(config_class, "description", ""))
    add_arguments(parser, all_attrs)
    args = parser.parse_args()
    if verbose:
        print("Parsed arguments:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
    flat_config_dict = vars(args)

    def build_instance(cls_type, prefix=""):
        instance = cls_type.__new__(cls_type)
        annotations = get_type_hints(cls_type)

        for attr, typ in annotations.items():
            full_key_prefix = f"{prefix}{attr}"
            # Handle nested config
            if inspect.isclass(typ) and hasattr(typ, "__annotations__"):
                nested_instance = build_instance(typ, prefix=f"{full_key_prefix}.")
                setattr(instance, attr, nested_instance)
            else:
                val = flat_config_dict.get(full_key_prefix, getattr(cls_type, attr, None))
                # Cast the value to the correct type if possible
                if val is not None and typ in [int, float, str, bool]:
                    try:
                        val = typ(val)
                    except Exception:
                        pass
                setattr(instance, attr, val)
        return instance

    return build_instance(config_class)
