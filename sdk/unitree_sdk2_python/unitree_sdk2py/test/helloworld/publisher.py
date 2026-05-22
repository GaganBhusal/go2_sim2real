import time

from sdk.unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from sdk.unitree_sdk2_python.unitree_sdk2py.test.helloworld.helloworld import HelloWorld

ChannelFactoryInitialize()

pub = ChannelPublisher("topic", HelloWorld)
pub.Init()

for i in range(30):
    msg = HelloWorld("Hello world. time:" + str(time.time()))
    # msg.data = "Hello world. time:" + str(time.time())

    if pub.Write(msg, 0.5):
        print("publish success. msg:", msg)
    else:
        print("publish error.")

    time.sleep(1)

pub.Close()