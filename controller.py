import pyfirmata

comport='COM12'

board=pyfirmata.Arduino(comport)

led_1 = board.get_pin('d:13:o')
led_2 = board.get_pin('d:12:o')
led_3 = board.get_pin('d:11:o')

def led(total):
    led_1.write(0)
    led_2.write(1)
    led_3.write(0)
