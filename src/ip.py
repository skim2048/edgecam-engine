import subprocess
import re


def get_local_ip(pattern="192\.168\."):
    result = subprocess.run(["ip", "addr"], stdout=subprocess.PIPE, text=True)
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
    for ip in ips:
        if re.match(pattern, ip):
            return ip
    return None
