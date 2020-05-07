import sys, time
from socket import *
argv = sys.argv

host = '127.0.0.1'
port = '12000'

timeout = 1

clientSocket = socket(AF_INET, SOCK_DGRAM)

clientSocket.settimeout(timeout)

port = int(port)

seq_num = 0

while seq_num < 10:
    try:
        seq_num += 1
        data = "Ping " + str(seq_num) + " " + time.asctime()
        RTTb = time.time()
        clientSocket.sendto(data.encode(), (host, port))
        message, address = clientSocket.recvfrom(1024)
        RTTa = time.time()
        print("Reply from " + address[0] + ": " + message.decode())
        RTT = RTTa - RTTb
        print("RTT: " + str(RTT))
    except:
        print("Request timed out")
clientSocket.close()