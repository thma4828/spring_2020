# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:26:36 2020

@author: tsmar
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:17:31 2020

@author: tsmar
"""

from socket import *
from datetime import date
port = 80

servsock = socket(AF_INET, SOCK_STREAM)

servsock.bind(('', port))

servsock.listen(1) #THE server creates a listening socket to listen for incoming http GET request, 

print('server ready')

class http_header_response:
    def __init__(self, version, server, type_, payload, connect, code, date, lmod):
        self.version = version
        self.server = server
        self.type = type_
        #self.code = 200
        self.connect = connect
        self.payload = payload
        self.payload_bytes = payload.encode()
        self.length_bytes = len(payload.encode())
        #self.date = ''
        #self.lmod = ''
        self.message = ''
        self.code = code
        self.date = date
        self.lmod = lmod
        
    def build_http(self):
        line1 = self.version + " " + self.code + "\r\n"
        line2 = "Connection: " + self.connect + "\r\n"
        line3 = "Date: " + self.date + "\r\n"
        line4 = "Server: " + self.server + "\r\n"
        line5 = "Last-Modified: " + self.lmod + "\r\n"
        line6 = "Content-Length: " + str(self.length_bytes)+ "\r\n"
        line7 = "Content-Type: " + self.type + "\r\n"
        line8 = self.payload + "\r\n" + "\r\n"
        self.message = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8
        
    def get_msg_bytes(self):
        return self.message.encode()
            
    
def get_date():
    return ""


while True:
    connectsock, addr = servsock.accept()
    recv = connectsock.recv(2048).decode()
    lines = recv.split(str="\r\n")
    if len(lines) < 1:
        print('not an http request, cannot be handled')
        connectsock.close()
    else:
        line_one = lines[0]
        line_one_fields = line_one.split(' ')
        if len(line_one_fields) < 3:
            print('not a http get request, cannot be handled')
            connectsock.close()
        elif line_one_fields[0] != 'GET':
            print('not a http get request, cannot be handled')
            connectsock.close()
        else:
            path = line_one_fields[1]
            date_ = get_date()
            version ="HTTP/1.0"
            server = "Python 3.7 (Windows)"
            response_type = "text/html"
            connect_type = "close"
            codeOK = "200 OK"
            codeNotFound = "404 Not Found"
            if len(path) > 1:
                path = path[1:] #get rid of slash assuming we are starting in local search path. 
            else:
                if path == '/':
                    path = 'homepage.html'
                else:
                    print("path invalid")
                    head = http_header_response(version, server, response_type, "", connect_type, codeNotFound, date, date)
                    
                    head.build_http()
                    
                    msg = head.get_msg_bytes()
                    
                    connectsock.send(msg)
                    connectsock.close()
                    exit(0)
            f = open(path, 'r')
            if f != None:
                    data = f.read() #any path given to the server will read data in, this is a security bug...
                  
                    head = http_header_response(version, server, response_type, data, connect_type, codeOK, date, date)
                    
                    head.build_http()
                    msg = head.get_msg_bytes()
                    
                    connectsock.send(msg)
            else:
                    head = http_header_response(version, server, response_type, "", connect_type, codeNotFound, date, date)
                    
                    head.build_http()
                    
                    msg = head.get_msg_bytes()
                    
                    connectsock.send(msg)
            f.close()
            connectsock.close()
    
    
