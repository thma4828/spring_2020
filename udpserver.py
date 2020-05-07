# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:10:46 2020

@author: tsmar
"""

from socket import *

port = 12000
servsock = socket(AF_INET, SOCK_DGRAM) #create a UDP socket, 
servsock.bind(('', port)) #bind socket to that port. 
print("server ready to recieve")
while True:
    message, addr = servsock.recvfrom(2048)
    msg = message.decode().upper()
    servsock.sendto(msg.encode(), addr)

