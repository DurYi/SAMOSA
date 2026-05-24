gpu=0
cd "$(dirname "$0")/.."

# LaSOT_ext
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_lasotext.py

# OTB100
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_otb100.py

# TrackingNet
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_trackingnet.py

# Anti-UAV300 test RGB
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_antiuav300.py

# Anti-UAV300 test infrared
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_antiuav300_ir.py

# Anti-UAV410 test
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_antiuav410.py

# Anti-UAV600 validation
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_antiuav600.py

# DUT Anti-UAV
PYTHONPATH="$PWD:$PWD/sam2:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=$gpu python scripts/test_dut.py
