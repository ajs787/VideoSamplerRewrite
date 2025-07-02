"""
WriteToDataset.py

This module provides functionality to write samples from a directory to a dataset tar file.

Functions:
    write_to_dataset(directory: str, tar_file: str, frames_per_sample: int = 1, out_channels: int = 1, batch_size: int = 10) -> None:
        Writes samples from a directory to a dataset tar file.

Dependencies:
    - webdataset
    - os
    - logging
    - time
    - torch
    - torchvision
    - io
    - concurrent.futures

Raises:
    Exception: If there is an error writing to the dataset.

License:#! /usr/bin/python3
"""
WriteToDataset.py

Writes valid samples into a WebDataset .tar, but preserves any truncated
or corrupted samples on disk for later debugging.
"""

import os
import io
import time
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor

import webdataset as wds
from PIL import Image

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)

def process_sample(key, png_root, txt_root, frames_per_sample, out_channels):
    """
    Reads one sample "<key>/" + "<key>.txt" and returns a WebDataset sample dict,
    or None if the folder doesn't contain exactly frames_per_sample PNGs or any
    of the PNGs are corrupt/truncated.
    """
    frame_dir = os.path.join(png_root, key)
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))

    # Check count
    if len(files) != frames_per_sample:
        logging.warning(
            f"{key}: expected {frames_per_sample} frames but found {len(files)}; leaving on disk"
        )
        return None

    # Verify each image is readable
    for fname in files:
        path = os.path.join(frame_dir, fname)
        try:
            with open(path, "rb") as f:
                data = f.read()
            # verify PNG integrity
            img = Image.open(io.BytesIO(data))
            img.verify()
        except Exception as e:
            logging.error(f"Truncated/corrupt image detected: {path} ({e}); dropping sample {key}")
            return None

    # build the sample dict
    sample = {}
    safe_key = key.replace(".", "_")
    sample["__key__"] = safe_key
    parts = key.split("_")
    if len(parts) < 2:
        logging.error(f"Malformed sample key: {key}")
        return None
    cls = parts[1]
    sample["cls"] = cls.encode("utf-8")

    # read metadata
    txt_path = os.path.join(txt_root, f"{key}.txt")
    try:
        with open(txt_path, "rb") as f:
            sample["metadata.txt"] = f.read()
    except Exception as e:
        logging.error(f"Could not read metadata for {key}: {e}")
        return None

    # read frames into memory
    for i, fname in enumerate(files):
        path = os.path.join(frame_dir, fname)
        try:
            with open(path, "rb") as img:
                sample[f"{i}.png"] = img.read()
        except Exception as e:
            logging.error(f"Could not read frame {path}: {e}")
            return None

    return sample


def write_to_dataset(
    png_root: str,
    tar_file: str,
    dataset_path: str,
    frames_per_sample: int = 1,
    out_channels: int = 3,
    batch_size: int = 60,
    equalize_samples: bool = False,
    max_workers: int = 4,
):
    """
    Walks each subfolder under png_root, writes *only* fully complete, uncorrupted samples
    into tar_file. Successful samples are deleted from disk; truncated/corrupt ones stay.
    """
    start = time.time()
    logging.info(f"Writing {tar_file} from samples in {png_root}")
    tar = wds.TarWriter(tar_file, encoder=False)
    txt_root = png_root.rstrip(os.sep) + "txt"
    keys = [d for d in os.listdir(png_root)
            if os.path.isdir(os.path.join(png_root, d))]
    logging.info(f"Found {len(keys)} sample folders")

    # optional equalization
    if equalize_samples:
        bycls = {}
        for k in keys:
            cls = k.split("_")[1]
            bycls.setdefault(cls, []).append(k)
        minc = min(len(v) for v in bycls.values())
        logging.info(f"Equalizing to {minc} samples per class")
        drop = []
        for v in bycls.values():
            v.sort()
            drop.extend(v[minc:])
        for d in drop:
            try:
                shutil.rmtree(os.path.join(png_root, d))
                os.remove(os.path.join(txt_root, f"{d}.txt"))
            except:
                pass
        keys = [k for k in keys if k not in drop]
        logging.info(f"Dropped {len(drop)} (equalize), {len(keys)} remain")

    count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            for key, sample in zip(
                batch,
                ex.map(
                    process_sample,
                    batch,
                    [png_root] * len(batch),
                    [txt_root] * len(batch),
                    [frames_per_sample] * len(batch),
                    [out_channels] * len(batch),
                ),
            ):
                if sample is None:
                    continue  # truncated/corrupt or error—left on disk

                # write valid sample to TAR
                tar.write(sample)
                count += 1

                # delete only successful samples
                try:
                    shutil.rmtree(os.path.join(png_root, key))
                    os.remove(os.path.join(txt_root, f"{key}.txt"))
                except Exception as e:
                    logging.warning(f"Cleanup failed for {key}: {e}")

                if count % 1000 == 0:
                    logging.info(f"  wrote {count} samples…")

    tar.close()
    logging.info(f"Finished writing {count}/{len(keys)} samples in {time.time()-start:.1f}s")

    with open(os.path.join(dataset_path, "RUN_DESCRIPTION.log"), "a+") as rd:
        rd.write(f"{count} samples → {tar_file}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("png_root")
    p.add_argument("tar_file")
    p.add_argument("dataset_path")
    p.add_argument("--frames_per_sample", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--equalize", action="store_true")
    p.add_argument("--max_workers", type=int, default=4)
    args = p.parse_args()
    write_to_dataset(
        args.png_root,
        args.tar_file,
        args.dataset_path,
        frames_per_sample=args.frames_per_sample,
        out_channels=args.out_channels,
        batch_size=args.batch_size,
        equalize_samples=args.equalize,
        max_workers=args.max_workers,
    )

    This project is licensed under the MIT License - see the LICENSE file for details.
"""
import datetime
import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import webdataset as wds
from torchvision import transforms


def process_sample(file, directory, frames_per_sample, out_channels):
    """

    :param file: param directory:
    :param frames_per_sample: param out_channels:
    :param directory: param out_channels:
    :param out_channels:

    """
    # convert the sample into something that can be read into the tar files
    try:
        data = np.load(os.path.join(directory, file))
        np_tensor = data["tensor"]
        frame = torch.from_numpy(np_tensor)

        s_c_file_path = os.path.join(directory + "txt",
                                     file.replace(".npz", ".txt"))
        with open(s_c_file_path, "r") as s_c_file:
            s = file.replace(".npz", "").split("/")[-1].split("_")
            if len(s) != 4:
                logging.error(
                    f"Unexpected format in file name: {file}, split result: {s}"
                )
                return None
            filename, d_name, _, _ = s
            video_path = filename.replace("SPACE", " ")
            sample_class = d_name
            frame_num = s_c_file.read().split("-")

        # to save space, immediately delete the sample's .npz and .txt file
        os.remove(os.path.join(directory, file))
        os.remove(s_c_file_path)

        base_name = os.path.basename(video_path).replace(" ", "_").replace(
            ".", "_")
        video_time = os.path.basename(video_path).split(".")[0]
        time_sec = time.mktime(time.strptime(video_time, "%Y-%m-%d %H:%M:%S"))
        time_struct = time.localtime(time_sec + int(frame_num[0]) // 3)
        curtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
        metadata = f"{video_path},{frame_num[0]},{curtime}"

        sample = {
            "__key__": "_".join((base_name, "_".join(frame_num))),
            "cls": str(sample_class).encode("utf-8"),
            "metadata.txt": metadata.encode("utf-8"),
        }

        # write sample / image to memory
        buffers = []
        for i in range(frames_per_sample):
            img = transforms.ToPILImage()(
                frame[i] / 255.0).convert(  # tar files are written as pngs
                    "RGB" if out_channels == 3 else "L")
            buf = io.BytesIO()  # saving the images to memory
            img.save(fp=buf, format="png")
            buffers.append(buf.getbuffer())

        for i, buffer in enumerate(buffers):
            sample[f"{i}.png"] = buffer

        return sample
    except RuntimeError as e:
        if "PytorchStreamReader" in str(e):
            # this is where the file is corrupted because the tensor wasn't read properly
            logging.error(
                f"PytorchStreamReader error processing sample {file}: {e}")
        else:
            logging.error(f"RuntimeError processing sample {file}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing sample {file}: {e}")
        return None


def write_to_dataset(
    directory: str,
    tar_file: str,
    dataset_path: str,
    frames_per_sample: int = 1,
    out_channels: int = 1,
    batch_size: int = 60,
    equalize_samples: bool = False,
    max_workers_tar_writing: int = 4,
):
    """Writes samples from a directory to a dataset tar file.

    :param directory: str
    :param tar_file: str
    :param frames_per_sample: int
    :param out_channels: int
    :param batch_size: int
    :param num_workers: int
    :param Raises: param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)
    :param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)
    :param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)
    :param directory: str:
    :param tar_file: str:
    :param dataset_path: str:
    :param frames_per_sample: int:  (Default value = 1)
    :param out_channels: int:  (Default value = 1)
    :param batch_size: int:  (Default value = 60)
    :param equalize_samples: bool:  (Default value = False)
    :param max_workers_tar_writing: int:  (Default value = 4)

    """
    try:
        tar_writer = wds.TarWriter(tar_file, encoder=False)
        start_time = time.time()

        file_list = [
            f for f in os.listdir(directory) if not f.endswith(".txt")
        ]

        # equalization to ensure the number of samples per class in each sample is
        # equal to each other (BUT NOT EQUALIZING SAMPLES ACROSS TAR FILES, THOSE
        # ARE INDEPENDENT)
        if equalize_samples:
            logging.info(f"Equalizing samples for {directory}")
            sample_dict = {}
            # first find the class with the least number of samples
            # then for each class, delete samples until the number
            # of samples is equal to the minimum
            for file in file_list:
                s = file.replace(".npz", "").split("/")[-1].split("_")
                _, sample_class, _, _ = s
                if sample_class in sample_dict:
                    sample_dict[sample_class].append(file)
                else:
                    sample_dict[sample_class] = [file]
            min_samples = min(
                [len(samples) for samples in sample_dict.values()])
            logging.info(
                f"Minimum number of samples for directory {directory}: {min_samples}"
            )
            for samples in sample_dict.values():
                random.shuffle(samples)
                for sample in samples[min_samples:]:
                    os.remove(os.path.join(directory, sample))
                    os.remove(
                        os.path.join(directory + "txt",
                                     sample.replace(".npz", ".txt")))
            logging.info(
                f"Equalized samples for {directory} and {directory + 'txt'}")

        file_list = [
            f for f in os.listdir(directory) if not f.endswith(".txt")
        ]
        file_size = len(file_list)
        logging.info(
            f"Reading in the samples from {directory}, finding {len(file_list)} files"
        )

        sample_count = 0  # for logging purposes
        old_time = time.time()
        # using threadpool because ilab is stingy with multiple processes
        # yes, I know about GIL lock
        with ThreadPoolExecutor(
                max_workers=max_workers_tar_writing) as executor:
            for i in range(0, len(file_list), batch_size):
                batch = file_list[i:i + batch_size]
                results = list(
                    executor.map(
                        # use batching here too, to speed up the process
                        process_sample,
                        batch,
                        [directory] * len(batch),
                        [frames_per_sample] * len(batch),
                        [out_channels] * len(batch),
                    ))
                for sample in results:
                    if sample:
                        tar_writer.write(sample)
                        sample_count += 1
                        if sample_count % 30000 == 0 and sample_count != 0:
                            new_time = time.time()
                            logging.info(
                                f"Writing sample {sample_count} to dataset tar file {tar_file}; time to write 30,000 samples: {((new_time - old_time)/30000):.2f} second(s) per sample"
                            )
                            old_time = new_time
        # make sure everything is written
        executor.shutdown(wait=True)
    except Exception as e:
        executor.shutdown(wait=False)
        logging.error(f"Error writing to dataset: {e}")
        raise e

    finally:
        # tar writer MUST CLOSE, or the data is unusable
        logging.info(f"Closing tar file {tar_file}")
        tar_writer.close()

    # logging into the RUN_DESCRIPTION
    with open(os.path.join(dataset_path, "RUN_DESCRIPTION.log"), "a+") as rd:
        rd.write(f"{file_size} samples collected by tar file {tar_file}\n")

    end_time = time.time()
    logging.info(
        f"Time taken to write to {tar_file}: {str(datetime.timedelta(seconds=int(end_time - start_time)))}"
    )
    logging.info(f"The number of samples in {tar_file}: {file_size}")
    return
