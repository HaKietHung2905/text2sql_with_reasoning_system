#!/usr/bin/env bash
# run_with_checkpoints.sh
# Runs generate_predictions.py in chunks of CHECKPOINT_SIZE,
# resuming automatically until all questions are done.
#
# Usage:
#   bash run_with_checkpoints.sh wikisql_full
#   bash run_with_checkpoints.sh wikisql_no_rb

set -euo pipefail

CONFIG=${1:-wikisql_full}           # label, controls which flags to pass
CHECKPOINT_SIZE=10
TOTAL=1000                          # set to actual number of questions
OUTPUT_DIR="results"
QUESTIONS="data/raw/wikisql/dev_spider_format.json"
DB="data/raw/wikisql/database"

# ── Map config label → flags ──────────────────────────────────────────────────
case "$CONFIG" in
  wikisql_full)
    OUTPUT="$OUTPUT_DIR/predictions_wikisql_full.tsv"
    FLAGS="--use_reasoning_bank --use_chromadb --use_semantic"
    ;;
  wikisql_no_rb)
    OUTPUT="$OUTPUT_DIR/predictions_wikisql_no_rb.tsv"
    FLAGS="--use_chromadb --use_semantic"
    ;;
  wikisql_no_rag)
    OUTPUT="$OUTPUT_DIR/predictions_wikisql_no_rag.tsv"
    FLAGS="--use_reasoning_bank"
    ;;
  wikisql_baseline)
    OUTPUT="$OUTPUT_DIR/predictions_wikisql_baseline.tsv"
    FLAGS=""
    ;;
  *)
    echo "Unknown config: $CONFIG"; exit 1
    ;;
esac

mkdir -p "$OUTPUT_DIR"

echo "=== Starting run: $CONFIG ==="
echo "    Output  : $OUTPUT"
echo "    Chunk   : $CHECKPOINT_SIZE questions per batch"
echo ""

ROUND=0
while true; do
    ROUND=$((ROUND + 1))

    # Count already-done lines
    DONE=0
    if [[ -f "$OUTPUT" ]]; then
        DONE=$(grep -c '' "$OUTPUT" 2>/dev/null || true)
    fi

    if [[ "$DONE" -ge "$TOTAL" ]]; then
        echo "✓ All $TOTAL questions done. Exiting."
        break
    fi

    REMAINING=$((TOTAL - DONE))
    echo "--- Round $ROUND | Done: $DONE / $TOTAL | Remaining: $REMAINING ---"

    RESUME_FLAG=""
    [[ "$DONE" -gt 0 ]] && RESUME_FLAG="--resume"

    # Refresh GCP credentials before each chunk
    export GOOGLE_CLOUD_ACCESS_TOKEN=$(gcloud auth print-access-token)

    python scripts/generate_predictions.py \
        --questions "$QUESTIONS" \
        --db        "$DB" \
        --output    "$OUTPUT" \
        --limit     "$TOTAL" \
        --checkpoint_size "$CHECKPOINT_SIZE" \
        $RESUME_FLAG \
        $FLAGS \
    || true  # don't abort on non-zero exit (500 errors etc.)

    echo "Chunk done. Sleeping 10s before next batch..."
    sleep 10
done

echo ""
echo "=== Run complete: $CONFIG ==="
echo "    Predictions: $OUTPUT"