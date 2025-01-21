import logging

logger = logging.getLogger('prophet')
logger.setLevel(logging.DEBUG)

logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('prophet.log', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

factory_id_set = {'1879092218430509051', '1879092218430509052', '1879092218430509053', '1879092218430509054', '1879092218430509055', '1879092218430509056'}
    
database_fetch_api_url = 'http://internal.elu-energy.com:800/api/system/vpp/load/data'
database_send_api_url = 'http://internal.elu-energy.com:800/api/system/vpp/load/syncPredictData'

STATION_ID = 'station_id'
DATETIME = 'datetime'
POWER = 'power'

INTERPOLATE_THRESHOLD = 8