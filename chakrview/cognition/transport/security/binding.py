"""
Peer Identity to TLS Certificate Binding (Step 31).

Establishes an explicit, policy-driven binding between transport-layer TLS certificates
and ChakrView application-layer CryptographicPeerIdentity instances.

CRITICAL ARCHITECTURAL AXIOMS:
1. TLS_IDENTITY != FEDERATION_AUTHORITY
2. mTLS != TRUST_GRANT
3. A valid TLS certificate for an unknown peer confers ZERO standing or registration.
4. Transport authentication and peer identity authentication are two separate gates.
"""

from typing import Dict, List, Optional, Tuple, Any

from chakrview.cognition.transport.security.models import (
    CertificateMetadata,
    PeerCertificateBinding,
)
from chakrview.cognition.transport.security.errors import (
    PeerBindingMismatchError,
)


class PeerCertificateBinder:
    """
    Manages explicit bindings between ChakrView peer IDs and TLS certificate fingerprints.
    """

    def __init__(self) -> None:
        self._bindings: Dict[str, PeerCertificateBinding] = {}

    def bind_peer(
        self,
        peer_id: str,
        certificate_fingerprint: str,
        expected_common_name: Optional[str] = None,
        expected_san: Optional[str] = None,
        epoch: int = 1,
    ) -> PeerCertificateBinding:
        """
        Register or update a peer-to-certificate binding.
        """
        binding = PeerCertificateBinding(
            peer_id=peer_id,
            certificate_fingerprint=certificate_fingerprint,
            expected_common_name=expected_common_name,
            expected_san=expected_san,
            binding_epoch=epoch,
            is_active=True,
        )
        self._bindings[peer_id] = binding
        return binding

    def get_binding(self, peer_id: str) -> Optional[PeerCertificateBinding]:
        """Lookup active binding for peer."""
        return self._bindings.get(peer_id)

    def unbind_peer(self, peer_id: str) -> bool:
        """Remove or deactivate binding."""
        if peer_id in self._bindings:
            del self._bindings[peer_id]
            return True
        return False

    def rotate_binding(
        self,
        peer_id: str,
        new_certificate_fingerprint: str,
        expected_common_name: Optional[str] = None,
        expected_san: Optional[str] = None,
        epoch: int = 1,
    ) -> PeerCertificateBinding:
        """
        Rotate a peer's bound certificate to a replacement fingerprint.
        """
        return self.bind_peer(
            peer_id=peer_id,
            certificate_fingerprint=new_certificate_fingerprint,
            expected_common_name=expected_common_name,
            expected_san=expected_san,
            epoch=epoch,
        )

    def verify_binding(
        self,
        peer_id: str,
        cert_meta: CertificateMetadata,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that the presented certificate matches the peer's bound identity.
        Returns: (is_valid, failure_reason)
        """
        binding = self._bindings.get(peer_id)
        if not binding:
            return False, f"No certificate binding exists for peer '{peer_id}'."

        if not binding.is_active:
            return False, f"Certificate binding for peer '{peer_id}' is inactive."

        if binding.certificate_fingerprint != cert_meta.fingerprint:
            return False, (
                f"Certificate fingerprint mismatch for peer '{peer_id}': "
                f"expected '{binding.certificate_fingerprint}', presented '{cert_meta.fingerprint}'."
            )

        if binding.expected_common_name is not None:
            actual_cn = cert_meta.subject.get("CN")
            if actual_cn != binding.expected_common_name:
                return False, (
                    f"Certificate CN mismatch for peer '{peer_id}': "
                    f"expected '{binding.expected_common_name}', got '{actual_cn}'."
                )

        if binding.expected_san is not None:
            if not cert_meta.matches_hostname(binding.expected_san):
                return False, (
                    f"Certificate SAN mismatch for peer '{peer_id}': "
                    f"expected SAN '{binding.expected_san}', not found in certificate."
                )

        return True, None

    def verify_binding_or_raise(
        self,
        peer_id: str,
        cert_meta: CertificateMetadata,
    ) -> None:
        """
        Verify binding and raise PeerBindingMismatchError if validation fails.
        """
        valid, reason = self.verify_binding(peer_id, cert_meta)
        if not valid:
            raise PeerBindingMismatchError(reason)

    def count(self) -> int:
        return len(self._bindings)
