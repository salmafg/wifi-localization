import argparse
import logging
import time

import AWSIoTPythonSDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

from config import MQTT

messages = []
AllowedActions = ['both', 'publish', 'subscribe']


def customCallback(client, userdata, message):
    """
    Custom MQTT message callback
    """
    # print("Received a new message: ")
    # print(message.payload)
    messages.append(message.payload)
    return messages


# Initialize parser
parser = argparse.ArgumentParser()

host = MQTT['endpoint']
rootCAPath = MQTT['rootCA']
certificatePath = MQTT['certificatePath']
privateKeyPath = MQTT['privateKeyPath']
port = MQTT['port']
useWebsocket = MQTT['useWebsocket']
clientId = MQTT['clientId']
topic = MQTT['topic']

if MQTT['mode'] not in AllowedActions:
    parser.error("Unknown --mode option %s. Must be one of %s" %
                 (MQTT['mode'], str(AllowedActions)))
    exit(2)

if MQTT['useWebsocket'] and MQTT['certificatePath'] and MQTT['privateKeyPath']:
    parser.error(
        "X.509 cert authentication and WebSocket are mutual exclusive. Please pick one.")
    exit(2)

if not MQTT['useWebsocket'] and (not MQTT['certificatePath'] or not MQTT['privateKeyPath']):
    parser.error("Missing credentials for authentication.")
    exit(2)

# Port defaults
# When no port override for WebSocket, default to 443
if MQTT['useWebsocket'] and not MQTT['port']:
    port = 443
# When no port override for non-WebSocket, default to 8883
if not MQTT['useWebsocket'] and not MQTT['port']:
    port = 8883

# Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.CRITICAL)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)

# Init AWSIoTMQTTClient
myAWSIoTMQTTClient = None
if useWebsocket:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(rootCAPath)
else:
    myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
    myAWSIoTMQTTClient.configureEndpoint(host, port)
    myAWSIoTMQTTClient.configureCredentials(
        rootCAPath, privateKeyPath, certificatePath)

# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
# Infinite offline Publish queueing
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec


myAWSIoTMQTTClient.connect()
if MQTT['mode'] == 'both' or MQTT['mode'] == 'subscribe':
    myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)


def get_messages():
    """
    Connect and subscribe to AWS IoT
    """
    time.sleep(3)
    global messages
    data = messages
    messages = []
    return data
