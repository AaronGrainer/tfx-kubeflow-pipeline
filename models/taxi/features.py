from typing import Text, List


DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']
BUCKET_FEATURE_KEYS = []
# BUCKET_FEATURE_KEYS = [
#   'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
#   'dropoff_longitude'
# ]
BUCKET_FEATURE_BUCKET_COUNT = [10, 10, 10, 10]

CATEGORICAL_FEATURE_KEYS = ['trip_start_hour',
                            'trip_start_day', 'trip_start_month']
CATEGORICAL_FEATURE_MAX_VALUES = [24, 31, 12]
VOCAB_FEATURE_KEYS = ['payment_type', 'company']

VOCAB_SIZE = 1000
OOV_SIZE = 10

LABEL_KEY = 'tips'


def transformed_name(key: Text) -> Text:
  return key + '_xf'


def vocabulary_name(key: Text) -> Text:
  return key + '_vocab'


def transformed_names(keys: List[Text]) -> List[Text]:
  return [transformed_name(key) for key in keys]
