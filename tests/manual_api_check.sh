#!/usr/bin/env bash
# Manual smoke test against a *running* deployment API container - not part of the
# pytest suite (see test_deployment_api.py for the automated, offline version of
# these same checks). Start the container first:
#   06_Deployment/build.sh && docker run --rm -p 8000:8000 skinbouncer-api:latest
#
# Usage:
#   tests/manual_api_check.sh [base_url] [player_name...]
#
# Defaults to http://localhost:8000 and checks a couple of well-known players;
# pass your own player names to check others.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
shift || true
if [ "$#" -eq 0 ]; then
    PLAYERS=(Notch jeb_)
else
    PLAYERS=("$@")
fi

echo "== GET / =="
curl -sf "$BASE_URL/" | jq .
echo

for player in "${PLAYERS[@]}"; do
    echo "== POST /check/player/ (player_name=$player) =="
    curl -sf -X POST "$BASE_URL/check/player/" \
        -H "Content-Type: application/json" \
        -d "{\"player_name\": \"$player\"}" | jq .
    echo
done

echo "== POST /check/player/ (empty identifiers -> expect 400) =="
curl -s -o /dev/null -w "status: %{http_code}\n" -X POST "$BASE_URL/check/player/" \
    -H "Content-Type: application/json" \
    -d '{"player_name": "", "player_id": ""}'
