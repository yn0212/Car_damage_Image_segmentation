"""car dataset."""

import tensorflow_datasets.public_api as tfds
from . import car_dataset_builder


class CarTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for nature dataset."""
  # TODO(nature):
  DATASET_CLASS = car_dataset_builder.My_10000_Car4
  SPLITS = {  # Expected number of examples on each split.
      "train": 6825,
      "test": 2925,
  }

  DL_EXTRACT_RESULT = {
      "images": ".",
      "annotations": ".",
  }


if __name__ == '__main__':
  tfds.testing.test_main()
