from src.utils import load_config, get_spark_session
from pyspark.sql.types import StringType
import logging as logger
from pyspark.sql import SparkSession


def preprocess_features(pdf, categorical_features, numerical_features, fillna_dict):
    """
    特征预处理核心逻辑：
    - 分类特征转字符串类型
    - 缺失值填充
    """
    # 分类特征转字符串
    for cat_col in categorical_features:
        pdf = pdf.withColumn(cat_col, pdf[cat_col].cast(StringType()))
    
    # 缺失值填充
    if fillna_dict:
        pdf = pdf.fillna(fillna_dict).fillna(0, subset=numerical_features)
    
    return pdf

def run_preprocess_job(spark: SparkSession):
    
    # 加载配置
    config = load_config("/Workspace/code/test_20260123/config/model_config.yaml")
    cat_features = config["features"]["categorical_features"]
    num_features = config["features"]["numerical_features"]
    fillna_dict = config["features"]["fillna_dict"]
    
    
    try:
        train_samp = spark.sql("SELECT * FROM db_sample.train_sample_temp")
        # 预处理
        train_samp_processed = preprocess_features(
            pdf=train_samp,
            categorical_features=cat_features,
            numerical_features=num_features,
            fillna_dict=fillna_dict
        )
        
        train_samp_processed.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable("db_sample.train_sample_preprocessed")
        
        logger.info(f"预处理完成 | 预处理后列名: {train_samp_processed.columns}")
        return train_samp_processed
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        raise e

if __name__ == "__main__":
    spark = get_spark_session()

    run_preprocess_job(spark)