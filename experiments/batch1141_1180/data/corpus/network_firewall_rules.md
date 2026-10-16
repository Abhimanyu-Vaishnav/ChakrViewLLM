# Firewall Configuration & Port Matrix

## 1. Permitted Inbound Traffic
- Inbound port 8443: Encrypted API traffic.
- Inbound port 9090: Prometheus telemetry scraping.
- Blocked ports: All outbound and inbound UDP traffic except port 5353 mDNS.
