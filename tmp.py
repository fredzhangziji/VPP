'''
用于分析3月份的异常高电价脚本
'''

import requests
url = 'https://api-pro-openet.terraqt.com/v1/gfs_surface/point'

headers = {
  'Content-Type': 'application/json',
  'token': 'lBDMwUjM2UzN2ADMwQWZwEjNzYjM2cDM'
}

data = {
  'lon': 120.02528586948561,
  'lat': 30.278024299726116,
  'mete_vars': ['d2m'],
  'time': '2025-03-31 00:00:00',
}

response = requests.request("POST", url, headers=headers, json=data)
print(response.json())
  