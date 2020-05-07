# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:39:49 2020

@author: tsmar
"""
from socket import *

mailhost = 'gmail.com'
userhost = '192.168.0.3'
rcptmail = 'thma4828@colorado.edu'
usermail = 'tsmarg@gmail.com'
msg = input('enter message to send: \n')
port = 25

clientsock = socket(AF_INET, SOCK_STREAM)

clientsock.connect((mailhost, port)) #make a TCP connection that is going to persist. 

#now need to do the handshake of SMTP


h_fields = clientsock.recv(2048).decode().split()

print(h_fields)

if h_fields[0] != "220" or h_fields[1] != mailhost:
    print("bad response from server")
    clientsock.close()
    exit(0)

r1 = "HELO " + userhost + " \r\n"

s1 = r1.encode()

clientsock.send(s1)

hello2 = clientsock.recv(2048).decode().split(' ')

print(hello2)

if hello2[0] != "250":
    print("bad response from server")
    clientsock.close()
    exit(0)
    
r2 = "MAIL FROM: <" + usermail + "> \r\n"
s2 = r2.encode()
clientsock.send(s2)

hello3 = clientsock.recv(2048).decode().split(' ')
print(hello3)

if hello3[0] != "250":
    print("bad response from server")
    clientsock.close()
    exit(0)
    
r3 = "RCPT TO: <" + rcptmail + "> \r\n"
s3 = r3.encode()

clientsock.send(s3)


hello4 = clientsock.recv(2048).decode().split(' ')
print(hello4)

if hello4[0] != "250":
    print("bad response from server")
    clientsock.close()
    exit(0)
    
s4 = "DATA \r\n".encode()
clientsock.send(s4)

hello5 = clientsock.recv(2048).decode().split(' ')
print(hello5)

if hello5[0] != "354":
    print("bad response from server")
    clientsock.close()
    exit(0)
    
mail = msg +  ' \r\n.\r\n'

mail_bytes = mail.encode()

clientsock.send(mail_bytes)

hello6 = clientsock.recv(2048).decode().split(' ')
print(hello6)

if hello6[0] != "250":
    print("bad response from server")
    clientsock.close()
    exit(0)

clientsock.send("QUIT \r\n".encode())

goodbye = clientsock.recv(2048).decode().split(' ')
print(goodbye)

if goodbye[0] != "221":
    print("bad response from server")
    clientsock.close()
    exit(0)

print("everything went as expected.")

clientsock.close()
exit(0)