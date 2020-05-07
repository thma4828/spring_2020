# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:21:23 2020

@author: tsmar
"""

from socket import *
port = 12000
host = 'hostname.com'

clientsock = socket(AF_INET, SOCK_STREAM) #sockstream is for TCP socket. 
clientsock.connect((host, port))

send = input('enter lowercase characters:')

clientsock.send(send.encode())

recv = clientsock.recv(1024)

print('From server: ', recv.decode())

clientsock.close()


