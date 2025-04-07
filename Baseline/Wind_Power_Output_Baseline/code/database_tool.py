from sqlalchemy import create_engine
import pandas as pd

# 数据库连接配置
DB_CONFIG = {
    'host': '10.5.0.10',
    'user': 'root',
    'password': 'kunyu2023rds',
    'database': 'vpp_service',
    'port': 3306
}

def get_db_connection():
    db_uri = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
    engine = create_engine(db_uri, echo=False)
    return engine

def read_from_db(query):
    engine = get_db_connection()
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def write_to_db(df, table_name, if_exists='append'):
    engine = get_db_connection()
    with engine.connect() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    print(f"数据已写入表 {table_name}.")

if __name__ == '__main__':
    # test read data
    query = "SELECT * FROM some_table LIMIT 10;"
    df_sample = read_from_db(query)
    print("读取数据示例:")
    print(df_sample.head())
    
    # test write data
    test_df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "B", "C"]
    })
    write_to_db(test_df, "test_table", if_exists='append')