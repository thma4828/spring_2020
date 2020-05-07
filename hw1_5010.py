# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:45:47 2020

@author: tsmar
"""

from socket import *
import sys

port = 80

serverSocket = socket(AF_INET, SOCK_STREAM)

serverSocket.bind(('', port))

serverSocket.listen(1)

while True:
    print("server ready to recieve")
    connectSocket, address = serverSocket.accept()
    try:
        message = connectSocket.recv(4096).decode()
        messageLines = message.split('\r\n')
        if len(message) < 1:
            continue
        print("message: ", message)
        
        lineOne = messageLines[0]
        print("request line: ", lineOne)
        lineOneFields = lineOne.split()
        
        assert(lineOneFields[0] == 'GET')
        print("type of req: ", lineOneFields[0])
        pathname = lineOneFields[1]
        pathname = pathname[1:]
        
        print("pathname: ", pathname)
        file = open(pathname, 'r')
        
        outputdata = file.read()
     
        line1 = 'HTTP/1.1 200 OK\r\n'
        line2 = 'Connection: close\r\n'
        line3 = 'Date: Tue, 4 Feb 2020 12:02:00\r\n'
        line4 = 'Server: Python/3.6 (Windows10)\r\n'
        line5 = 'Last-Modified: Tue, 4 Feb 2020 12:02:00\r\n'
        line6 = 'Content-Length: ' + str(len(outputdata.encode())) + '\r\n'
        line7 = 'Content-Type: text/html\r\n'
        
        header = line1 + line2 + line3 + line4 + line5 + line6 + line7
        
        for byte in header:
            connectSocket.send(byte.encode())
            
        connectSocket.send('\r\n'.encode()) #send extra CRLF to signify message body on the way. 
            
        for byte in outputdata:
            connectSocket.send(byte.encode())
    
        connectSocket.close()
        
    except IOError:
        print('file not found on server')
        lineone = 'HTTP/1.1 404 Not Found\r\n\r\n'
        connectSocket.send(lineone.encode())
        connectSocket.close()

serverSocket.close()
sys.exit()