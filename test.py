import requests


def send(dialogue):
    res = requests.post('http://localhost:6000/', json=dialogue)
    return res.json()['data']


dialogue = []
while True:
    val = input()
    dialogue.append(val)
    res = send(dialogue)
    print(res)
    dialogue.append(res)