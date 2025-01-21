import util.constant as const
import util.data_cleansing_util as data_util
import util.prophet_util as prophet
import pickle

const.logger.info('Start to proceed single day data predictions.')

for factory_id in const.factory_id_set:
    prophet.predict_one_day(factory_id, None)