build: clean
    python setup_ops.py install
clean:
    pip uninstall paddle-sparse-ops -y
test: build
    pytest -s -v