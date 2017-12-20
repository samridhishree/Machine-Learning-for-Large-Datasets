#!/usr/bin/env python

if __name__ == '__main__':
  
  import argparse
  import os
  import time
  
  script_dir = os.path.dirname(os.path.realpath(__file__))
  default_host_file = os.path.join(script_dir, 'hosts.txt')
  
  parser = argparse.ArgumentParser(description='Kills the JBosen LDA application.')
  parser.add_argument('--host_file', type=str, default=default_host_file, help='Path to the host file to use.')
  parser.add_argument('--pem_file', type=str, default='',help='Location of AWS pem file')
  args = parser.parse_args()

  with open(args.host_file, 'r') as f:
    host_ips = [line.split(':')[0] for line in f]

  for client_id, ip in enumerate(host_ips):
    if args.pem_file:
        aws_args = "-i " + args.pem_file
    else:
        aws_args = " "
    cmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ' + aws_args + " " + ip + ' '
    cmd += '\'pkill -f ".*jython.jar.*"\''
    print(cmd)
    os.system(cmd)

