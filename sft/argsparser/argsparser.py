import argparse
from typing import Dict, Any, Optional
import yaml


class ArgsProcessor:
    def __init__(self, config_path: str) -> None:
        """
        Initialize ArgsProcessor with a configuration file path.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            None
        """
        self.config_path: str = config_path

    def flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Recursively flattens a nested dictionary, but does not add the parent key.
        
        Args:
            d (Dict[str, Any]): Input dictionary to flatten
            parent_key (str, optional): Parent key (unused in this implementation). Defaults to ''
            sep (str, optional): Separator for nested keys. Defaults to '_'
            
        Returns:
            Dict[str, Any]: Flattened dictionary
        """
        items: list = []
        for k, v in d.items():
            new_key: str = k  # Use the current key directly, without adding the parent key
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def add_args_from_yaml(self, args: argparse.Namespace) -> argparse.Namespace:
        """
        Add contents of YAML configuration file to args object.

        Args:
            args (argparse.Namespace): Argument namespace to update

        Returns:
            argparse.Namespace: Updated argument namespace
        """
        # Read the YAML configuration file
        with open(self.config_path, 'r') as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Process configuration with support for nested structures
        processed_config = self._process_config(config)

        # Add the configuration items to args
        for key, value in processed_config.items():
            setattr(args, key, value)

        return args

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process configuration dictionary, preserving nested structures for specific keys
        while flattening others.

        Args:
            config: Raw configuration dictionary

        Returns:
            Processed configuration dictionary
        """
        # Keys that should preserve their nested structure
        nested_keys = {'dynamic_length', 'optimization', 'special_tokens', 'expansion_decision'}

        processed = {}

        for key, value in config.items():
            if key in nested_keys and isinstance(value, dict):
                # Preserve nested structure for specific keys
                processed[key] = self._convert_types(value)
            elif isinstance(value, dict):
                # Flatten other nested dictionaries
                flat_dict = self.flatten_dict(value)
                converted_flat = self._convert_types(flat_dict)
                if isinstance(converted_flat, dict):
                    processed.update(converted_flat)
            else:
                # Direct assignment for non-dict values
                processed[key] = self._convert_single_value(value)

        return processed

    def _convert_types(self, data):
        """
        Recursively convert string values to appropriate types.

        Args:
            data: Data to convert (dict, list, or single value)

        Returns:
            Converted data
        """
        if isinstance(data, dict):
            return {k: self._convert_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_types(item) for item in data]
        else:
            return self._convert_single_value(data)

    def _convert_single_value(self, value):
        """
        Convert a single value to appropriate type.

        Args:
            value: Value to convert

        Returns:
            Converted value
        """
        if isinstance(value, str):
            # Convert boolean strings
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            # Convert float strings
            elif 'e' in value or '.' in value:
                try:
                    return float(value)
                except ValueError:
                    pass
        return value