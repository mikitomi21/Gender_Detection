from pathlib import Path
import shutil
import kaggle


class DataDownloader:
    @staticmethod
    def download_data(skip_if_exists: bool = True):
        # TODO wszedzie popieram .gitkeep
        DataDownloader._prepare_dirs()
        DataDownloader._download_babies_images(skip_if_exists)
        DataDownloader._download_genders_images(skip_if_exists)
        DataDownloader._download_none_images(skip_if_exists)

    @staticmethod
    def _prepare_dirs():
        DATA_PATH = Path(__file__).parent.parent / "data"

        # raw dirs
        raw_directory = DATA_PATH / "raw"
        raw_directory.mkdir(parents=True, exist_ok=True)

        babies_directory = DATA_PATH / "raw" / "babies"
        babies_directory.mkdir(parents=True, exist_ok=True)

        # processed dirs
        processed_directory = DATA_PATH / "processed"
        processed_directory.mkdir(parents=True, exist_ok=True)

        train_directory = DATA_PATH / "processed" / "train"
        train_directory.mkdir(parents=True, exist_ok=True)
        DataDownloader._prepare_classes_dirs(train_directory)

        test_directory = DATA_PATH / "processed" / "test"
        test_directory.mkdir(parents=True, exist_ok=True)
        DataDownloader._prepare_classes_dirs(test_directory)

        val_directory = DATA_PATH / "processed" / "val"
        val_directory.mkdir(parents=True, exist_ok=True)
        DataDownloader._prepare_classes_dirs(val_directory)

    @staticmethod
    def _prepare_classes_dirs(path: Path):
        babies_directory = path / "babies"
        babies_directory.mkdir(parents=True, exist_ok=True)

        men_directory = path / "men"
        men_directory.mkdir(parents=True, exist_ok=True)

        women_directory = path / "women"
        women_directory.mkdir(parents=True, exist_ok=True)

        none_directory = path / "none"
        none_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _download_babies_images(skip_if_exists: bool):
        BABIES_DATASET = "frabbisw/facial-age"
        RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw"
        PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
        BABIES_DATA_PATH = RAW_DATA_PATH / "babies"
        FACE_AGE_DATA_PATH = RAW_DATA_PATH / "face_age"

        if skip_if_exists and any(
            file.is_file() for file in BABIES_DATA_PATH.iterdir()
        ):
            print(f"🙈 Dataset '{BABIES_DATASET}' is already downloaded.")
            return

        kaggle.api.dataset_download_files(
            dataset=BABIES_DATASET, path=str(RAW_DATA_PATH), unzip=True
        )

        selected_years = ["001", "002", "003", "004"]
        for year in selected_years:
            baby_year = FACE_AGE_DATA_PATH / year
            for file in baby_year.iterdir():
                shutil.copy(file, BABIES_DATA_PATH)

        number_of_images = len(
            [file for file in BABIES_DATA_PATH.iterdir() if file.is_file()]
        )
        TRAIN_SIZE = int(number_of_images * 2 / 3)
        TEST_SIZE = int((number_of_images - TRAIN_SIZE) / 2)
        for i, file in enumerate(BABIES_DATA_PATH.iterdir()):
            if i <= TRAIN_SIZE:
                shutil.copy(file, PROCESSED_DATA_PATH / "train" / "babies")
            elif TRAIN_SIZE < i < TRAIN_SIZE + TEST_SIZE:
                shutil.copy(file, PROCESSED_DATA_PATH / "test" / "babies")
            else:
                shutil.copy(file, PROCESSED_DATA_PATH / "val" / "babies")

        shutil.rmtree(FACE_AGE_DATA_PATH)

        print(
            f"✅ Dataset '{BABIES_DATASET}' downloaded to directory: {str(BABIES_DATA_PATH)}"
        )

    @staticmethod
    def _download_genders_images(skip_if_exists: bool):
        GENDERS_DATASET = "ashishjangra27/gender-recognition-200k-images-celeba"
        PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
        TEST_MEN_PATH = PROCESSED_DATA_PATH / "test" / "men"

        if skip_if_exists and any(file.is_file() for file in TEST_MEN_PATH.iterdir()):
            print(f"🙈 Dataset '{GENDERS_DATASET}' is already downloaded.")
            return

        kaggle.api.dataset_download_files(
            dataset=GENDERS_DATASET, path=str(PROCESSED_DATA_PATH), unzip=True
        )

        temp_dir = PROCESSED_DATA_PATH / "Dataset"
        common_dirs = [
            ["Test/Male", "test/men"],
            ["Test/Female", "test/women"],
            ["Train/Male", "train/men"],
            ["Train/Female", "train/women"],
            ["Validation/Male", "val/men"],
            ["Validation/Female", "val/women"],
        ]
        for src_dir, dst_dir in common_dirs:
            full_src_dir = temp_dir / src_dir
            full_dst_dir = PROCESSED_DATA_PATH / dst_dir
            for file in full_src_dir.iterdir():
                shutil.move(str(file), str(full_dst_dir / file.name))

        shutil.rmtree(temp_dir)

        print(
            f"✅ Dataset '{GENDERS_DATASET}' downloaded to directory: {str(PROCESSED_DATA_PATH)}"
        )

    @staticmethod
    def _download_none_images(skip_if_exists: bool):
        NONE_DATASET = "felicepollano/watermarked-not-watermarked-images"
        PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
        TEST_NONE_PATH = PROCESSED_DATA_PATH / "test" / "none"

        if skip_if_exists and any(file.is_file() for file in TEST_NONE_PATH.iterdir()):
            print(f"🙈 Dataset '{NONE_DATASET}' is already downloaded.")
            return

        kaggle.api.dataset_download_files(
            dataset=NONE_DATASET, path=str(PROCESSED_DATA_PATH), unzip=True
        )

        temp_dir = PROCESSED_DATA_PATH / "wm-nowm"
        common_dirs = [
            ["train/no-watermark", "train/none"],
            ["valid/no-watermark", "val/none"],
        ]
        for src_dir, dst_dir in common_dirs:
            full_src_dir = temp_dir / src_dir
            val_dst_dir = PROCESSED_DATA_PATH / dst_dir
            test_dst_dir = PROCESSED_DATA_PATH / "test/none"
            for i, file in enumerate(full_src_dir.iterdir()):
                if "val" in str(dst_dir) and i % 2 == 0:
                    shutil.move(str(file), str(test_dst_dir / file.name))
                else:
                    shutil.move(str(file), str(val_dst_dir / file.name))

        shutil.rmtree(temp_dir)

        print(
            f"✅ Dataset '{NONE_DATASET}' downloaded to directory: {str(PROCESSED_DATA_PATH)}"
        )
