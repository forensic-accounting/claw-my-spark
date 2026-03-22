#!/usr/bin/env bash
# approve-device.sh — interactive OpenClaw device approval
set -euo pipefail

SPARK="claude@dgx-spark-claude"
CONTAINER="openclaw"

# Restore terminal on exit (handles Ctrl-C mid-read)
restore_term() { tput cnorm; stty echo 2>/dev/null || true; }
trap restore_term EXIT

echo "Fetching pending requests..."

raw=$(ssh "$SPARK" "docker exec $CONTAINER openclaw devices list 2>&1")

# Parse request IDs and display info from the Pending section only
declare -a ids=()
declare -a labels=()

while IFS= read -r line; do
    if [[ "$line" =~ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}) ]]; then
        uuid="${BASH_REMATCH[1]}"
        role=$(echo "$line" | awk -F'│' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}')
        age=$(echo "$line"  | awk -F'│' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $6); print $6}')
        ids+=("$uuid")
        labels+=("$uuid  [role: $role]  $age")
    fi
done < <(echo "$raw" | awk '/^Pending/,/^Paired/')

if [ ${#ids[@]} -eq 0 ]; then
    echo "No pending device requests."
    echo ""
    echo "$raw"
    exit 0
fi

selected=0
count=${#ids[@]}

draw() {
    for i in "${!ids[@]}"; do
        if [ "$i" -eq "$selected" ]; then
            printf "  \033[1;32m▶  %s\033[0m\n" "${labels[$i]}"
        else
            printf "     %s\n" "${labels[$i]}"
        fi
    done
}

erase() {
    for _ in "${ids[@]}"; do tput cuu1; tput el; done
}

echo ""
echo "↑↓ to select   y to approve   q to quit"
echo ""
draw

tput civis  # hide cursor
stty -echo

while true; do
    IFS= read -rsn1 k
    if [[ $k == $'\x1b' ]]; then
        IFS= read -rsn2 -t 0.05 k2 || true
        k+="$k2"
    fi
    case "$k" in
        $'\x1b[A')  # up
            erase
            (( selected = (selected - 1 + count) % count ))
            draw
            ;;
        $'\x1b[B')  # down
            erase
            (( selected = (selected + 1) % count ))
            draw
            ;;
        y|Y)
            stty echo; tput cnorm
            echo ""
            echo "Approving ${ids[$selected]} ..."
            ssh "$SPARK" "docker exec $CONTAINER openclaw devices approve -- ${ids[$selected]}"
            echo "Approved."
            exit 0
            ;;
        q|Q|$'\x03')
            stty echo; tput cnorm
            echo ""
            echo "Cancelled."
            exit 0
            ;;
    esac
done
