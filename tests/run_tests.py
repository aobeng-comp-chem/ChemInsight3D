import subprocess
import sys

cmd = [sys.executable, '-m', 'pytest', '-q', 'tests/test_localization_native.py']
print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, check=False)
raise SystemExit(result.returncode)
