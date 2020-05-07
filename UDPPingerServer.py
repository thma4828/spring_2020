import random
from socket import *

print("hello server")

serverSocket = socket(AF_INET, SOCK_DGRAM)

serverSocket.bind(('', 12000))

while True:
    print("in while loop")
    value = random.randint(0, 10)
    message, address = serverSocket.recvfrom(1024)
    message = message.decode()
    message = message.upper()
    print("message: ", message)
    if value < 4:
        continue
    else:
        serverSocket.sendto(message.encode(), address)