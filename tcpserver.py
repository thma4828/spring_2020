# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:17:31 2020

@author: tsmar
"""

from socket import *
port = 12000
host = 'thhost.com'

servsock = socket(AF_INET, SOCK_STREAM)

servsock.bind(('', port))

servsock.listen(1)

print('server ready')

while True:
    connectsock, addr = servsock.accept()
    recv = connectsock.recv(1024).decode()
    send = recv.upper()
    connectsock.send(send.encode())
    connectsock.close()
    
