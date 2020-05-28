import os
from absl import logging
import config as pipeline_config
from pipeline import config
from pipeline import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.proto import trainer_pb2


def run():
  metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
  tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)
  runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=metadata_config,
    tfx_image=tfx_image
  )

  os.environ[kubeflow_dag_runner.SDK_ENV_LABEL] = 'tfx-template'

  kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    pipeline.create_pipeline(
      pipeline_name=config.PIPELINE_NAME,
      pipeline_root=pipeline_config.PIPELINE_ROOT_GCS,
      data_path=pipeline_config.DATA_PATH_KUBEFLOW,
      preprocessing_fn=config.PREPROCESSING_FN,
      run_fn=config.RUN_FN,
      train_args=trainer_pb2.TrainArgs(num_steps=config.TRAIN_NUM_STEPS),
      eval_args=trainer_pb2.EvalArgs(num_steps=config.EVAL_NUM_STEPS),
      eval_accuracy_threshold=config.EVAL_ACCURACY_THRESHOLD,
      serving_model_dir=pipeline_config.SERVING_MODEL_DIR_GCS,
      # query=config.BIG_QUERY_QUERY,
      # beam_pipeline_args=config.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
      # beam_pipeline_args=config.DATAFLOW_BEAM_PIPELINE_ARGS,
      # ai_platform_training_args=config.GCP_AI_PLATFORM_TRAINING_ARGS,
      # ai_platform_serving_args=config.GCP_AI_PLATFORM_SERVING_ARGS
    ))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
