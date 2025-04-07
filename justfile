clean:
    pip uninstall paddle-sparse-ops -y
build: clean
    python setup_ops.py install
test: build
    pytest -s -v