# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:03:18 2020

@author: tsmar
"""

from socket import *
 server = 'hostname:'
 port = 12000
 
 server = server + str(port)
 
 clientsock = socket(AF_INET, SOCK_DGRAM) #creates the client socket
 #AF_INET means ipv4, SOCK_DGRAM means UDP socket
 
 message = input('input msg in lowercase:')
 
 #convert the message from string to bytes. 
 clientsock.sendto(message.encode(), (server, port))
 #so the message bytes are the payload that gets loaded into a UDP packet... 
 
 msg, addr = clientsock.recvfrom(2048) #packet data put into msg, packet source put into addr
 #2048 is size of input
 #the variable server has bot the IP and p port number
 
 print(msg.decode())

clientsock.close()
 
 