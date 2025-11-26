set -e
python -m venv .venv
source .venv/bin/activate
make install
make