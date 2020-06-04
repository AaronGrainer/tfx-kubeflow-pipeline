from typing import Optional, Text, List, Dict, Any
import tensorflow_model_analysis as tfma

from ml_metadata.proto import metadata_store_pb2
from tfx.components import BigQueryExampleGen
from tfx.components import CsvExampleGen
from tfx.components import ImportExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    data_path: Text,
    preprocessing_fn: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    eval_accuracy_threshold: float,
    serving_model_dir: Text,
    query: Optional[Text] = None,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:

  if query:
    example_gen = BigQueryExampleGen(query=query)
  else:
    # example_gen = CsvExampleGen(input=external_input(data_path))
    example_gen = ImportExampleGen(input=external_input(data_path))

  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                         infer_feature_shape=False)

  example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'],
                                       schema=schema_gen.outputs['schema'])

  transform = Transform(examples=example_gen.outputs['examples'],
                        schema=schema_gen.outputs['schema'],
                        preprocessing_fn=preprocessing_fn)

  trainer_args = {
    'run_fn': run_fn,
    'transformed_examples': transform.outputs['transformed_examples'],
    'schema': schema_gen.outputs['schema'],
    'transform_graph': transform.outputs['transform_graph'],
    'train_args': train_args,
    'eval_args': eval_args,
    'custom_executor_spec':
        executor_spec.ExecutorClassSpec(
          trainer_executor.GenericExecutor),
  }
  if ai_platform_training_args:
    trainer_args.update({
      'custom_executor_spec':
        executor_spec.ExecutorClassSpec(
            ai_platform_trainer_executor.GenericExecutor),
      'custom_config': {
        ai_platform_trainer_executor.TRAINING_ARGS_KEY:
          ai_platform_training_args,
      }
    })
  trainer = Trainer(**trainer_args)

  # model_resolver = ResolverNode(instance_name='latest_blessed_model_resolver',
  #                               resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
  #                               model=Channel(type=Model),
  #                               model_blessing=Channel(type=ModelBlessing))

  # eval_config = tfma.EvalConfig(
  #     model_specs=[tfma.ModelSpec(label_key='tips')],
  #     slicing_specs=[tfma.SlicingSpec()],
  #     metrics_specs=[
  #         tfma.MetricsSpec(
  #             thresholds={
  #                 'binary_accuracy':
  #                 tfma.config.MetricThreshold(
  #                     value_threshold=tfma.GenericValueThreshold(
  #                         lower_bound={'value': eval_accuracy_threshold}),
  #                     change_threshold=tfma.GenericChangeThreshold(
  #                         direction=tfma.MetricDirection.HIGHER_IS_BETTER,
  #                         absolute={'value': -1e-10}))
  #             })
  #     ])
  # evaluator = Evaluator(examples=example_gen.outputs['examples'],
  #                       model=trainer.outputs['model'],
  #                       baseline_model=model_resolver.outputs['model'],
  #                       eval_config=eval_config)

  # pusher_args = {
  #   'model':
  #     trainer.outputs['model'],
  #   'model_blessing':
  #     evaluator.outputs['blessing'],
  #   'push_destination':
  #     pusher_pb2.PushDestination(
  #       filesystem=pusher_pb2.PushDestination.Filesystem(
  #         base_directory=serving_model_dir)),
  # }
  # if ai_platform_serving_args:
  #   pusher_args.update({
  #     'custom_executor_spec': 
  #       executor_spec.ExecutorClassSpec(
  #         ai_platform_pusher_executor.Executor),
  #     'custom_config': {
  #       ai_platform_pusher_executor.SERVING_ARGS_KEY:
  #         ai_platform_serving_args
  #     },
  #   })
  # pusher = Pusher(**pusher_args)

  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=[
      example_gen,
      statistics_gen,
      schema_gen,
      example_validator,
      # transform,
      # trainer,
      # model_resolver,
      # evaluator,
      # pusher
    ],
    enable_cache=True,
    metadata_connection_config=metadata_connection_config,
    beam_pipeline_args=beam_pipeline_args,
  )
