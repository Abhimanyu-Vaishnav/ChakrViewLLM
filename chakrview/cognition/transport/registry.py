"""
Transport Adapter Registry (Step 30).

Provides centralized lookup and factory dispatch for physical and logical transport adapters.
"""

from typing import Dict, Type, Optional

from chakrview.cognition.transport.base import Transport
from chakrview.cognition.transport.loopback import LoopbackWireTransport
from chakrview.cognition.transport.tcp import TCPWireTransport
from chakrview.cognition.transport.http2 import HTTP2WireTransport
from chakrview.cognition.transport.grpc import GRPCWireTransport
from chakrview.cognition.transport.errors import TransportUnavailableError


class TransportRegistry:
    """
    Central registry mapping transport scheme names to implementation classes.
    """

    _REGISTRY: Dict[str, Type[Transport]] = {
        "loopback": LoopbackWireTransport,
        "tcp": TCPWireTransport,
        "http2": HTTP2WireTransport,
        "grpc": GRPCWireTransport,
    }

    @classmethod
    def register(cls, scheme: str, transport_cls: Type[Transport]) -> None:
        """Register custom or specialized transport driver."""
        cls._REGISTRY[scheme.lower()] = transport_cls

    @classmethod
    def get(cls, scheme: str) -> Type[Transport]:
        """Retrieve transport driver class by scheme name."""
        transport_cls = cls._REGISTRY.get(scheme.lower())
        if not transport_cls:
            raise TransportUnavailableError(
                f"No transport driver registered for scheme '{scheme}'. Available: {list(cls._REGISTRY.keys())}"
            )
        return transport_cls

    @classmethod
    def create(cls, endpoint: str) -> Transport:
        """
        Factory creating transport instance based on endpoint scheme prefix.
        e.g. 'tcp://127.0.0.1:9000' -> TCPWireTransport
             'loopback://peer1' -> LoopbackWireTransport
        """
        if "://" in endpoint:
            scheme = endpoint.split("://")[0].lower()
        else:
            scheme = "tcp" if ":" in endpoint else "loopback"

        transport_cls = cls.get(scheme)
        return transport_cls(endpoint=endpoint)

    @classmethod
    def list_supported_schemes(cls) -> Dict[str, bool]:
        """List registered transport schemes and their current runtime availability."""
        results = {}
        for scheme, transport_cls in cls._REGISTRY.items():
            if hasattr(transport_cls, "is_available"):
                try:
                    inst = transport_cls()
                    results[scheme] = bool(getattr(inst, "is_available", True))
                except Exception:
                    results[scheme] = False
            else:
                results[scheme] = True
        return results
