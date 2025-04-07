import tool

cities = [
    '呼和浩特',
    '阿拉善盟']

single_results = tool.test_single_city(cities, day_index=-1)
province_actual, province_forecast, province_metrics = tool.predict_province_day(cities, day_index=-1)