#!/usr/bin/env bash
# =============================================================================
# expressvpn/entrypoint.sh — OpenVPN sidecar for OpenClaw
# Connects to ExpressVPN and preserves Docker bridge routing so openclaw
# can still reach ollama while all internet traffic goes through the VPN.
# =============================================================================
set -euo pipefail

USERNAME="${EXPRESS_VPN_OPENVPN_USERNAME:?EXPRESS_VPN_OPENVPN_USERNAME is required}"
PASSWORD="${EXPRESS_VPN_OPENVPN_PASSWORD:?EXPRESS_VPN_OPENVPN_PASSWORD is required}"
CONFIG_FILE="${EXPRESS_VPN_CONFIG:?EXPRESS_VPN_CONFIG is required}"
CONFIG_SRC="/etc/openvpn/configs/${CONFIG_FILE}"
CONFIG_WORK="/tmp/vpn.ovpn"
AUTH_FILE="/tmp/vpn-auth.txt"

# ── 1. Credentials ────────────────────────────────────────────────────────────
echo "[vpn] Writing credentials..."
printf '%s\n%s\n' "$USERNAME" "$PASSWORD" > "$AUTH_FILE"
chmod 600 "$AUTH_FILE"

# ── 2. Patch config for OpenVPN 2.6 compatibility ────────────────────────────
# ns-cert-type was removed in OpenVPN 2.5; replace with remote-cert-tls.
# Inject auth-user-pass file path so credentials are read automatically.
echo "[vpn] Preparing config from ${CONFIG_FILE}..."
sed \
  -e 's/ns-cert-type server/remote-cert-tls server/' \
  -e "s|^auth-user-pass$|auth-user-pass ${AUTH_FILE}|" \
  "$CONFIG_SRC" > "$CONFIG_WORK"

# ── 3. Capture Docker gateway before VPN changes routing ─────────────────────
ORIG_GW=$(ip route show default | awk '/default/{print $3; exit}')
echo "[vpn] Docker gateway: ${ORIG_GW}"

# ── 4. Connect ────────────────────────────────────────────────────────────────
connect() {
    echo "[vpn] Killing any existing openvpn processes..."
    pkill openvpn 2>/dev/null || true
    sleep 1
    echo "[vpn] Starting OpenVPN..."
    openvpn --config "$CONFIG_WORK" --daemon openvpn --log /tmp/openvpn.log
}

# ExpressVPN uses redirect-gateway def1 which adds 0.0.0.0/1 and 128.0.0.0/1
# routes via tun rather than replacing the default route. Check for those.
vpn_is_up() {
    ip route show 0.0.0.0/1 2>/dev/null | grep -q tun
}

wait_for_vpn() {
    echo "[vpn] Waiting for VPN routes..."
    for i in $(seq 1 60); do
        if vpn_is_up; then
            echo "[vpn] VPN routes are up."
            return 0
        fi
        sleep 1
    done
    echo "[vpn] ERROR: VPN routes did not appear after 60s. OpenVPN log:"
    cat /tmp/openvpn.log
    return 1
}

# ── 5. Restore Docker bridge routes after VPN changes default route ───────────
restore_docker_routes() {
    echo "[vpn] Restoring Docker bridge routes via ${ORIG_GW}..."
    # 172.16.0.0/12 covers all default Docker bridge/compose networks
    ip route replace 172.16.0.0/12 via "$ORIG_GW" metric 100
    # 10.0.0.0/8 covers the host LAN
    ip route replace 10.0.0.0/8    via "$ORIG_GW" metric 100
}

connect
wait_for_vpn
restore_docker_routes
echo "[vpn] Connected. Monitoring..."

# ── 6. Monitor and reconnect on disconnect ────────────────────────────────────
while true; do
    sleep 30
    if ! vpn_is_up; then
        echo "[vpn] VPN route lost, reconnecting..."
        connect
        sleep 10
        wait_for_vpn && restore_docker_routes
    fi
done
