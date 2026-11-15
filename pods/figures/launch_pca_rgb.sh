OUTPUT_DIR=${1:-/workspace/eval/pca_rgb}

python -u /workspace/gif-7005_apprentissage_automatique/pods/figures/make_pca_rgb_figure.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --concepts Mass "Suspicious Calcification" "Focal Asymmetry" --num-images 2 --grid 48 28 \
  --output-dir "$OUTPUT_DIR"
