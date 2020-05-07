# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:47:45 2020

@author: tsmar
"""

from socket import *
from datetime import date
port = 80
hostname = "somehost.org"
path = "/resfile.html"
path_home = "/"
method = "GET"
version = "HTTP/1.0"
connect = "close"
user_agent = "Mozilla/5.0"
lang = "en"
class html_header_get:
    def __init__(self, method, path, version, host, connect, agent, lang):
        self.line_1 = method + " " + path + " " + version + "\r\n"
        self.line_2 = "Host: " + host + "\r\n"
        self.line_3 = "Connection: " + connect + "\r\n"
        self.line_4 = "User-agent: " + agent + "\r\n"
        self.line_5 = "Accept-language: " + lang + "\r\n" + "\r\n"
    
    def get_http(self):
        self.full = self.line_1 + self.line_2 + self.line_3 + self.line_4 + self.line_5
        return self.full 
    
    def get_bytes(self):
        return self.full.encode()
        
host = "thehost.com"

clientsock = socket(AF_INET, SOCK_STREAM) #sockstream is for TCP socket. 
clientsock.connect((host, port))

html = html_header_get(method, path, version, hostname, connect, user_agent, lang)

html.get_http()

byts = html.get_bytes()

clientsock.send(byts)

recv = clientsock.recv(2048).decode()

lines = recv.split("\r\n")
#now you would have to parse the HTTP response to see if it succeeded or not. 
if len(lines) < 1:
    print("error: no valid response from server.")
    exit(0)

line1 = lines[0]

line1_fields = line1.split(' ')
n_ = len(line1_fields)
if n_ < 3 or n_ > 3:
    print("error: no valid response from server.")
    exit(0)
    
if line1_fields[1] == "200": #response OK
    n = len(line1_fields) - 1 #this may not be right, data may be after the double CRLF! 
    data = line1_fields[n] #should probably be the last line huh... 
    file = open("clientres1.txt", 'w+') #saves file, then browswer could interpret it. 
    file.write(data)
    print(data)
    file.close()
else:
    print("response failed with code: ", line1_fields[1])
    exit(0)
    

    
