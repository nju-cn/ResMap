sudo -E PATH="$PATH" PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))") cgexec -g cpu,cpuacct:mylim /usr/bin/python3 main.py "$@"