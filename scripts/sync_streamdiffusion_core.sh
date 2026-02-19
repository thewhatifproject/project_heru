#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REPO="https://github.com/chenfengxu714/StreamDiffusionV2.git"
TMP_DIR="${TMPDIR:-/tmp}/streamdiffusionv2-sync-$$"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$ROOT_DIR/core/streamdiffusionv2"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

git clone "$UPSTREAM_REPO" "$TMP_DIR"

mkdir -p "$CORE_DIR/causvid/models" "$CORE_DIR/streamv2v" "$CORE_DIR/configs"

cp "$TMP_DIR/LICENSE" "$CORE_DIR/LICENSE"
cp "$TMP_DIR/README.md" "$CORE_DIR/UPSTREAM_README.md"
cp "$TMP_DIR/requirements.txt" "$CORE_DIR/upstream-requirements.txt"
cp "$TMP_DIR/configs/wan_causal_dmd_v2v.yaml" "$CORE_DIR/configs/"
cp "$TMP_DIR/configs/wan_causal_dmd_v2v_14b.yaml" "$CORE_DIR/configs/"
cp "$TMP_DIR/configs/wan_causal_dmd_warp_4step_cfg2.yaml" "$CORE_DIR/configs/"
cp "$TMP_DIR/causvid/scheduler.py" "$CORE_DIR/causvid/"
cp -R "$TMP_DIR/causvid/models/wan" "$CORE_DIR/causvid/models/"
cp "$TMP_DIR/causvid/models/model_interface.py" "$CORE_DIR/causvid/models/"
cp -R "$TMP_DIR/streamv2v/communication" "$CORE_DIR/streamv2v/"
cp "$TMP_DIR/streamv2v/inference.py" "$CORE_DIR/streamv2v/"
cp "$TMP_DIR/streamv2v/inference_wo_batch.py" "$CORE_DIR/streamv2v/"

rm -f "$CORE_DIR/causvid/models/wan/bidirectional_inference.py"
rm -f "$CORE_DIR/causvid/models/wan/causal_inference.py"
rm -f "$CORE_DIR/causvid/models/wan/generate_ode_pairs.py"
rm -f "$CORE_DIR/streamv2v/communication/test_communication.py"

# Make TextDataset imports lazy so the runtime can import without dataset helpers.
perl -0pi -e 's/from causvid\.data import TextDataset\n//g' "$CORE_DIR/streamv2v/inference.py"
perl -0pi -e 's/\n    dataset = TextDataset\(args\.prompt_file_path\)/\n    from causvid.data import TextDataset  # Optional dependency for CLI mode\\n\\n    dataset = TextDataset(args.prompt_file_path)/g' "$CORE_DIR/streamv2v/inference.py"

perl -0pi -e 's/from causvid\.data import TextDataset\n//g' "$CORE_DIR/streamv2v/inference_wo_batch.py"
perl -0pi -e 's/\n    dataset = TextDataset\(args\.prompt_file_path\)/\n    from causvid.data import TextDataset  # Optional dependency for CLI mode\\n\\n    dataset = TextDataset(args.prompt_file_path)/g' "$CORE_DIR/streamv2v/inference_wo_batch.py"

# Keep local integration glue files untouched if present.
if [[ ! -f "$CORE_DIR/causvid/__init__.py" ]]; then
  cat > "$CORE_DIR/causvid/__init__.py" <<'PYEOF'
"""StreamDiffusionV2 core namespace (vendored subset)."""
PYEOF
fi

if [[ ! -f "$CORE_DIR/streamv2v/__init__.py" ]]; then
  cat > "$CORE_DIR/streamv2v/__init__.py" <<'PYEOF'
"""Core StreamV2V modules vendored from StreamDiffusionV2."""
PYEOF
fi

if [[ ! -f "$CORE_DIR/causvid/data.py" ]]; then
  cat > "$CORE_DIR/causvid/data.py" <<'PYEOF'
from __future__ import annotations

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Minimal prompt dataset used by streamv2v CLI entrypoints."""

    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as handle:
            self.texts = [line.strip() for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]
PYEOF
fi

echo "Synced StreamDiffusionV2 core subset into: $CORE_DIR"

echo "Upstream commit: $(git -C "$TMP_DIR" rev-parse HEAD)"
echo "Upstream date:   $(git -C "$TMP_DIR" show -s --format='%ci' HEAD)"
