import time

from sdk.unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from sdk.unitree_sdk2_python.example.helloworld.user_data import *


if __name__ == "__main__":
    ChannelFactoryInitialize()
    # Create a subscriber to subscribe the data defined in UserData class
    sub = ChannelSubscriber("topic", UserData)
    sub.Init()

    while True:
        msg = sub.Read()
        if msg is not None:
            print("Subscribe success. msg:", msg)
        else:
            print("No data subscribed.")
            break
    sub.Close()
