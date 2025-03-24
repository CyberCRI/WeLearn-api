#! /bin/bash

set -euo pipefail

if test -d /secrets; then
    for secretFile in /secrets/*; do
        secretName=$(basename "$secretFile")
        secretValue=$(cat "$secretFile")
        echo "Setting environment variable: $secretName"
        export "$secretName=$secretValue"
    done
else
    echo "No secrets directory found, skipping."
fi

exec "$@"
