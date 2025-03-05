#!/usr/bin/python3

import paramiko
import os
import random
import argparse
from threading import Thread
from tqdm import trange
import time

def randomString(stringLength=16):
    """Generate a random hex-string of fixed length """
    return ''.join(chr(random.randrange(256)).encode("ISO-8859-1").hex() for _ in range(stringLength))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmdFile', type=str, help="Path to a file containing the commands to launch.")
    parser.add_argument('-c', type=int, default=100, dest='count', help='Number of traces')
    args = parser.parse_args()

    # Parse command file
    cmdLines = []
    with open(args.cmdFile) as f:
        for line in f:
            if len(line) > 1:
                cmdLines.append(line.strip("\n").split(","))

    # Initialize SSH connection
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.128.37', username='ocarlon', password='TFG34')
    ssh_transp = ssh.get_transport()

    for _ in trange(args.count):
        for pre, cmd, tag in cmdLines:
            dataFile = "trace-{}.dat".format(randomString(16))
            print("Executing: {}".format(cmd))
            if pre:
                chan = ssh_transp.open_session()
                chan.exec_command(pre)
                while not chan.exit_status_ready():
                    time.sleep(0.01)
                chan.close()
            # Launch command
            chan = ssh_transp.open_session()
            chan.exec_command(cmd)
            chan.close()
    
    ssh.close()
