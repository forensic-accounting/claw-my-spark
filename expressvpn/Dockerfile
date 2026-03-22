FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      openvpn \
      iproute2 \
      iptables \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/openvpn/configs

COPY configs/ /etc/openvpn/configs/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
