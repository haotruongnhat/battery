# -*- coding:utf-8 -*-
import RPi.GPIO as GPIO
import serial

ser = serial.Serial("/dev/ttyTHS1",115200,timeout=10) 
print("open serial port: %s" %ser.portstr)

sbuf = "RS485 test...\r\n"
print("send: %s" %sbuf)  
len = ser.write(sbuf) 
print("send len = %d" %len)

rbuf = ""
while(1):
	rbuf = ser.readline()
	if(rbuf != ""):
		print(rbuf)
		# print("I send back what I received:")
		sbuf = "RS485 Received success!\r\n"
		ser.write(sbuf)
		rbuf = ""
	

print("--------------------------------------------------------")

ser.flush()


