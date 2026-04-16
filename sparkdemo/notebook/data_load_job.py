from src.utils import load_config, get_spark_session,find_repo_root
import logging as logger
import os
from pyspark.sql import SparkSession

def run_data_load_job(spark: SparkSession):
    
    # 加载配置
    config = load_config("/Workspace/code/test_20260123/config/model_config.yaml")
    csv_file_name = config["data"]["csv_file_name"]
    target_column = config["data"]["target_column"]
    
    # 获取项目根目录 + 拼接CSV文件路径
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = "/Workspace/code/test_20260123/"
    csv_file_path = os.path.join(project_root, csv_file_name)
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_file_path} | 请在项目根目录放置 {csv_file_name}")
    
  


    try:

        train_samp = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(csv_file_path)
        )
        
        # 筛选特征+目标列
        feature_cols = config["features"]["categorical_features"] + config["features"]["numerical_features"]

        if target_column not in train_samp.columns:
            raise ValueError(f"❌ CSV文件中缺少目标列: {target_column} | 现有列: {train_samp.columns}")
        
        missing_cols = [col for col in feature_cols if col not in train_samp.columns]

        if missing_cols:
            raise ValueError(f"❌ CSV文件中缺少特征列: {missing_cols} | 现有列: {train_samp.columns}")
        
        train_samp = train_samp.select(feature_cols + [target_column])
        
        train_samp.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable("db_sample.train_sample_temp")
        

        
        return train_samp
    except Exception as e:
        logger.error(f"❌ CSV文件加载失败: {str(e)}")
        raise e

if __name__ == "__main__":
    # 本地测试
    spark = get_spark_session()

    run_data_load_job(spark)