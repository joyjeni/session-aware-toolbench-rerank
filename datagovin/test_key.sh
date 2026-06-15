#!/usr/bin/env bash
# Usage: ./test_key.sh <YOUR_API_KEY>
# Tests data.gov.in API key against three popular resources.

set -e
KEY="${1:?Provide your API key as the first argument}"
BASE="https://api.data.gov.in/resource"

echo "Key: ${KEY:0:8}...${KEY: -4}  length=${#KEY}"
echo

for RID in \
  "35985fb7-f9e1-450b-b8c1-25571bc0fc89" \
  "9ef84268-d588-465a-a308-a864a43d0070" \
  "6176ee09-3d56-4a3b-8115-21841576b2f6" ; do
  echo "--- resource $RID ---"
  curl -sS -w "HTTP %{http_code}  time=%{time_total}s\n" \
    "$BASE/$RID?api-key=$KEY&format=json&limit=1" \
    | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('HTTP')[0] or '{}'); print('keys:', list(d.keys())[:8]); print('records:', len(d.get('records',[])))" 2>/dev/null || true
  echo
done
