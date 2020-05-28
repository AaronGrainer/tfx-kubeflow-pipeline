import os

from pipeline import config


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

OUTPUT_DIR = os.path.join(".", "tfx")
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             config.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', config.PIPELINE_NAME,
                             'metadata.db')

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")


OUTPUT_DIR_GCS = os.path.join('gs://', config.GCS_BUCKET_NAME)
PIPELINE_ROOT_GCS = os.path.join(OUTPUT_DIR_GCS, 'tfx_pipeline_output',
                                 config.PIPELINE_NAME)
SERVING_MODEL_DIR_GCS = os.path.join(PIPELINE_ROOT_GCS, 'serving_model')

DATA_PATH_KUBEFLOW = 'data'
