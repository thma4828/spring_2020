# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:52:11 2020

@author: tsmar
"""

from socket import *
import sys
import threading

def targ(connectSocket):
    print('new connection thread created')
    try:
        message = connectSocket.recv(4096).decode()
        messageLines = message.split('\r\n')
        print("message: ", message)
        if len(message) < 1:
            return
        lineOne = messageLines[0]
        print("request line: ", lineOne)
        lineOneFields = lineOne.split()
        
        assert(lineOneFields[0] == 'GET')
        print("type of req: ", lineOneFields[0])
        pathname = lineOneFields[1]
        pathname = pathname[1:]
        
        if '?' in pathname:
            paths = pathname.split('?')
            print("pathname: ", paths[0])
            arglist = paths[1]
            argvals = arglist.split('&')
            arg_value_array = []
            for argv in argvals:
                f = argv.split('=')
                name = f[0]
                value = f[1]
                arg_value_array.append((name, value))
            if paths[0] == 'simpleresponse.html':
                fname = arg_value_array[0]
                lname = arg_value_array[1]
                first_name = fname[1].strip()
                last_name = lname[1].strip()
                print("server visited by: ", first_name, " ", last_name)
            pathname = paths[0]
        
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


def main_f():
    port = 80

    serverSocket = socket(AF_INET, SOCK_STREAM)

    serverSocket.bind(('', port))

    serverSocket.listen(1)
    threads = []
    while True:
        print('listener thread ready')
        connectSocket, address = serverSocket.accept() 
        conn_thread = threading.Thread(target=targ, args=([connectSocket]))
        conn_thread.start()
        threads.append(conn_thread)
        
        if len(threads) > 5:
            for thread in threads:
                thread.join()
                
            threads = []
    serverSocket.close()


listener = threading.Thread(target=main_f, args=())
listener.start()
listener.join()
sys.exit()