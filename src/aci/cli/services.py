"""
Services container module for ACI CLI.

This module provides backward compatibility by re-exporting
ServicesContainer and create_services from aci.services.container.

For new code, prefer importing directly from aci.services.container
or aci.services.
"""

# Re-export from centralized container module for backward compatibility
from aci.services.container import ServicesContainer, create_services

__all__ = [
    "ServicesContainer",
    "create_services",
]
