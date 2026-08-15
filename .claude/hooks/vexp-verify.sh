#!/bin/bash
# vexp-verify: mechanical completion gate on Stop (Horizon). Fails open.
VEXP_BIN="c:/Users/donal/.vscode/extensions/vexp.vexp-vscode-2.6.2-win32-x64/binaries/vexp-core-win32-x64/vexp-core.exe"
[ -x "$VEXP_BIN" ] || exit 0
"$VEXP_BIN" stop-gate 2>/dev/null
exit 0
