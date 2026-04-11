"""Configuration management utilities."""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager with dot notation access."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                Path(__file__).parent.parent.parent, 
                "config", 
                "config.yaml"
            )
        
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access for top-level keys."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> "Config":
        """Update configuration with new values."""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(self._config, updates)
        return self
    
    def save(self, path: str = None):
        """Save configuration to YAML file."""
        save_path = path or self.config_path
        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)


_config_instance = None


def get_config(config_path: str = None) -> Config:
    """Get or create global configuration instance."""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance