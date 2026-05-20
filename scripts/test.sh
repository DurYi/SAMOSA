gpu=0
cd "$(dirname "$0")/.."

# Anti-UAV300 test RGB
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_antiuav300.py

# Anti-UAV300 test infrared
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_antiuav300_ir.py

# More datasets will be updated soon — stay tuned!