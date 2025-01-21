import util.constant as const
import util.data_cleansing_util as data_util
import util.prophet_util as prophet

for factory_id in const.factory_id_set:
    json_data = data_util.fetch_single_factory_data(factory_id)
    df_data = data_util.transfer_response_data(json_data)

    df_cleansed_data = data_util.cleanse_single_factory_data(df_data)

    df_normed_data = data_util.normalize_data(df_cleansed_data)
    
    model = prophet.train(df_normed_data)
    prophet.predict_all(factory_id, model)