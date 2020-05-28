import os
from absl import logging

import config as pipeline_config
from pipeline import config
from pipeline import pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import trainer_pb2


def run():
  BeamDagRunner().run(
    pipeline.create_pipeline(
      pipeline_name=config.PIPELINE_NAME,
      pipeline_root=pipeline_config.PIPELINE_ROOT,
      data_path=pipeline_config.DATA_PATH,
      preprocessing_fn=config.PREPROCESSING_FN,
      run_fn=config.RUN_FN,
      train_args=trainer_pb2.TrainArgs(num_steps=config.TRAIN_NUM_STEPS),
      eval_args=trainer_pb2.EvalArgs(num_steps=config.EVAL_NUM_STEPS),
      eval_accuracy_threshold=config.EVAL_ACCURACY_THRESHOLD,
      serving_model_dir=pipeline_config.SERVING_MODEL_DIR,
      # query=config.BIG_QUERY_QUERY,
      # beam_pipeline_args=config.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          pipeline_config.METADATA_PATH)
    ))


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()
