import time

from sdk.unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from sdk.unitree_sdk2_python.unitree_sdk2py.test.helloworld.helloworld import HelloWorld

ChannelFactoryInitialize()

sub = ChannelSubscriber("topic", HelloWorld)
sub.Init()

while True:
    msg = sub.Read()

    if msg is None:
        print("subscribe error.")
    else:
        print("subscribe success. msg:", msg)

pub.Close()