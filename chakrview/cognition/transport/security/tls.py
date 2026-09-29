"""
TLS Context Construction and Configuration (Step 31).

Constructs hardened server and client SSLContext objects based on SecureTransportPolicy.
Enforces:
1. TLS 1.3 preference (or explicitly permitted TLS 1.2)
2. Obsolete protocols (SSLv2, SSLv3, TLS 1.0, TLS 1.1) are permanently disabled
3. Mandatory certificate verification by default
4. mTLS mutual authentication (CERT_REQUIRED for client certs)
5. Zero leakage of private key material in logs or errors
"""

import os
import ssl
import tempfile
from typing import Dict, List, Optional, Any

from chakrview.cognition.transport.security.models import (
    TLSMode,
    TLSProtocolVersion,
)
from chakrview.cognition.transport.security.policy import SecureTransportPolicy
from chakrview.cognition.transport.security.errors import (
    TLSError,
    TLSConfigurationError,
    InsecureDowngradeError,
)


class TLSContextFactory:
    """
    Factory creating secure, policy-compliant server and client SSLContext instances.
    """

    @staticmethod
    def _apply_version_policy(ctx: ssl.SSLContext, policy: SecureTransportPolicy) -> None:
        """Enforce minimum TLS protocol version on context."""
        policy.validate()

        if policy.minimum_tls_version == TLSProtocolVersion.TLS_1_3:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        elif policy.minimum_tls_version == TLSProtocolVersion.TLS_1_2:
            if not policy.allow_tls_1_2:
                raise TLSConfigurationError("TLS 1.2 is disabled by policy (allow_tls_1_2 is False).")
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        else:
            raise TLSConfigurationError(f"Unsupported TLS minimum version: {policy.minimum_tls_version}")


    @staticmethod
    def _load_cert_and_key_pem(ctx: ssl.SSLContext, cert_pem: str, key_pem: str) -> None:
        """
        Safely load certificate and private key from PEM strings into SSLContext.
        Uses ephemeral temporary files that are unlinked immediately after loading.
        """
        c_fd, c_path = tempfile.mkstemp(prefix="cv_cert_", suffix=".pem")
        k_fd, k_path = tempfile.mkstemp(prefix="cv_key_", suffix=".pem")
        try:
            with open(c_fd, "w", encoding="utf-8") as cf:
                cf.write(cert_pem)
            with open(k_fd, "w", encoding="utf-8") as kf:
                kf.write(key_pem)

            ctx.load_cert_chain(certfile=c_path, keyfile=k_path)
        except Exception as e:
            raise TLSConfigurationError(f"Failed to load certificate and private key chain: {e}")
        finally:
            if os.path.exists(c_path):
                os.unlink(c_path)
            if os.path.exists(k_path):
                os.unlink(k_path)

    @staticmethod
    def create_server_context(
        policy: SecureTransportPolicy,
        cert_pem: str,
        key_pem: str,
        ca_cert_pem: Optional[str] = None,
    ) -> ssl.SSLContext:
        """
        Create a hardened server-side SSLContext.
        """
        policy.validate()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        TLSContextFactory._apply_version_policy(ctx, policy)

        # Load server certificate & private key
        TLSContextFactory._load_cert_and_key_pem(ctx, cert_pem, key_pem)

        # Handle client verification (mTLS vs standard TLS)
        if policy.tls_mode == TLSMode.MTLS or policy.require_client_certificate:
            ctx.verify_mode = ssl.CERT_REQUIRED
            # CA configuration is mandatory for mTLS
            ca_loaded = False
            if ca_cert_pem:
                ctx.load_verify_locations(cadata=ca_cert_pem)
                ca_loaded = True
            if policy.trusted_ca_data:
                ctx.load_verify_locations(cadata=policy.trusted_ca_data)
                ca_loaded = True
            for path in policy.trusted_ca_paths:
                ctx.load_verify_locations(cafile=path)
                ca_loaded = True

            if not ca_loaded:
                raise TLSConfigurationError(
                    "Mutual TLS (mTLS) server context requires at least one trusted CA source."
                )
        else:
            ctx.verify_mode = ssl.CERT_NONE

        return ctx

    @staticmethod
    def create_client_context(
        policy: SecureTransportPolicy,
        ca_cert_pem: Optional[str] = None,
        cert_pem: Optional[str] = None,
        key_pem: Optional[str] = None,
    ) -> ssl.SSLContext:
        """
        Create a hardened client-side SSLContext.
        """
        policy.validate()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        TLSContextFactory._apply_version_policy(ctx, policy)

        # Certificate verification policy
        if policy.verify_peer_certificate:
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.check_hostname = policy.verify_hostname

            ca_loaded = False
            if ca_cert_pem:
                ctx.load_verify_locations(cadata=ca_cert_pem)
                ca_loaded = True
            if policy.trusted_ca_data:
                ctx.load_verify_locations(cadata=policy.trusted_ca_data)
                ca_loaded = True
            for path in policy.trusted_ca_paths:
                ctx.load_verify_locations(cafile=path)
                ca_loaded = True

            # If no custom CA was provided, load default system verify locations
            if not ca_loaded:
                ctx.load_default_certs()
        else:
            if not policy.allow_insecure_test_downgrade:
                raise InsecureDowngradeError(
                    "Disabling peer certificate verification in client context requires allow_insecure_test_downgrade=True."
                )
            ctx.verify_mode = ssl.CERT_NONE
            ctx.check_hostname = False

        # Load client certificate for mTLS if provided
        if cert_pem and key_pem:
            TLSContextFactory._load_cert_and_key_pem(ctx, cert_pem, key_pem)
        elif policy.tls_mode == TLSMode.MTLS:
            raise TLSConfigurationError("Client context for mTLS requires cert_pem and key_pem.")

        return ctx
