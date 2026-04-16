from src.utils import load_config, get_spark_session
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from synapse.ml.lightgbm import LightGBMClassifier
import pandas as pd
import logging as logger
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession


def lightgbm_classification_training(train_pdf, hyperparameter_dictionary, categorical_features, numerical_features, mlflow_config):

    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
        for col in categorical_features
    ]

    indexers2 = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
        for col in numerical_features
    ]
    
    feature_cols = [f"{col}_index" for col in numerical_features] + [f"{col}_index" for col in categorical_features]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    model = LightGBMClassifier(**hyperparameter_dictionary)
    
    stages = indexers+ indexers2 + [assembler, model]
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(train_pdf)
    
    lgb_model = pipelineModel.stages[-1]
    feature_importances = lgb_model.getFeatureImportances()
    fi_series = pd.Series(feature_importances, index=feature_cols)
    fi_dict = fi_series.sort_values(ascending=False).to_dict()
    
    if mlflow_config["enable_log"]:
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config["experiment_name"])
        with mlflow.start_run():
            mlflow.spark.log_model(pipelineModel, "lightgbm_model")
            mlflow.log_dict(fi_dict, "feature_importance.json")
            mlflow.log_params(hyperparameter_dictionary)
    
    return pipelineModel, fi_dict


def run_train_job(spark: SparkSession):

    
    # 加载配置
    config = load_config("/Workspace/code/test_20260123/config/model_config.yaml")
    model_params = config["model"]["params"]
    cat_features = config["features"]["categorical_features"]
    num_features = config["features"]["numerical_features"]
    mlflow_config = config["mlflow"]
    
    
    # 读取预处理数据
    try:
        train_samp_processed = spark.sql("SELECT * FROM db_sample.train_sample_preprocessed")

        train_samp_processed_columns = train_samp_processed.columns
        logger.info(f'tran_samp_columns: {train_samp_processed_columns}')
        
        # 训练模型
        pipelineModel, feature_importance = lightgbm_classification_training(
            train_pdf=train_samp_processed,
            hyperparameter_dictionary=model_params,
            categorical_features=cat_features,
            numerical_features=num_features,
            mlflow_config=mlflow_config
        )
        
        model_save_path = "/Workspace/data/jars/spark_lightgbm_model"
        # pipelineModel.write().overwrite().save(model_save_path)
        logger.info(f"✅ 模型训练完成 | 模型已保存至: {model_save_path}")
        
        return pipelineModel, feature_importance
    except Exception as e:
        logger.error(f" 模型训练失败: {str(e)}")
        raise e

if __name__ == "__main__":
    spark = get_spark_session()

    run_train_job(spark)