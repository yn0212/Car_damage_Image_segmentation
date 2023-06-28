
import os

from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_LABEL_CLASSES = ["Breakage", "Scratched","Crushed","Separated"]

_SPECIES_CLASSES = ["Breakage", "Scratched","Crushed","Separated"]


class My_10000_Car4(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version("0.1.0")

  def _info(self):
    return self.dataset_info_from_configs(

        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(),
            "label": tfds.features.ClassLabel(names=_LABEL_CLASSES),
            "species": tfds.features.ClassLabel(names=_SPECIES_CLASSES),
            "file_name": tfds.features.Text(),
            "segmentation_mask": tfds.features.Image(
                shape=(None, None, 1), use_colormap=True
            ),
        }),
        supervised_keys=("image", "label"),
    )

  def _split_generators(self, dl_manager):
    """Returns splits."""

    dl_paths = dl_manager.download_and_extract({
        "images": 'D:/jyn/ncslab/data/car/images.tar.gz',
        "annotations":'D:/jyn/ncslab/data/car/annotations.tar.gz',
    })

    images_path_dir = os.path.join(dl_paths["images"], "images")
    annotations_path_dir = os.path.join(dl_paths["annotations"], "annotations")


    train_split = tfds.core.SplitGenerator(
        name="train",
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(
                annotations_path_dir, "trainval.txt"
            ),
        },
    )
    test_split = tfds.core.SplitGenerator(
        name="test",
        gen_kwargs={
            "images_dir_path": images_path_dir,
            "annotations_dir_path": annotations_path_dir,
            "images_list_file": os.path.join(annotations_path_dir, "test.txt"),
        },
    )

    return [train_split, test_split]

  def _generate_examples(
      self, images_dir_path, annotations_dir_path, images_list_file
  ):
    with tf.io.gfile.GFile(images_list_file, "r") as images_list:
      for idx,line in enumerate(images_list):
        image_name, label, species, _ = line.strip().split(" ")

        trimaps_dir_path = os.path.join(annotations_dir_path, "trimaps")

        trimap_name = image_name + ".png"
        image_name += ".jpg"
        label = int(label) - 1
        species = int(species) - 1

        record = {
            "image": os.path.join(images_dir_path, image_name),
            "label": int(label),
            "species": species,
            "file_name": image_name,
            "segmentation_mask": os.path.join(trimaps_dir_path, trimap_name),
        }
        yield idx , record
