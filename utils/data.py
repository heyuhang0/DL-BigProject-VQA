import io
import json
import os
import zipfile
from typing import Callable, Iterable, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, download_url


def yes_no_only_filter(q: str, a: str) -> bool:
    return a == 'yes' or a == 'no'


def single_word_answer_filter(q: str, a: str) -> bool:
    return len(a.split()) == 1


class VQA2Dataset(Dataset):
    """ VQA2 Dataset """

    resources = [
        # (URL, extract?)
        ('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip', True),
        ('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip', True),
        ('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip', True),
        ('https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip', True),
        ('http://images.cocodataset.org/zips/train2014.zip', False),
        ('http://images.cocodataset.org/zips/val2014.zip', False)
    ]

    def __init__(
            self,
            root: str,
            train: bool = True,
            filter_func: Callable[[str, str], bool] = yes_no_only_filter,
            image_transform: Optional[Callable] = None,
            question_transform: Optional[Callable] = None,
            question_transform_factory: Optional[Callable[[Iterable[str]], Callable]] = None,
            answer_transform: Optional[Callable] = None,
            answer_transform_factory: Optional[Callable[[Iterable[str]], Callable]] = None,
            download: bool = False
    ) -> None:
        """
        Args:
            root (string): Root directory of dataset
            train (bool, optional): If True, creates the train set,
                otherwise creates the validation set
            image_transform (callable, optional): Optional transform to be applied on the input PIL image.
            question_transform (callable, optional): Optional transform to be applied on the input question string.
            question_transform_factory (callable, optional): Optional tranform factory to override text_transform;
                the factory function should take in training corpus(list of string) and return a text_transform
                function.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        super().__init__()
        self.root = root
        self.train = train
        self.image_transform = image_transform
        self.question_transform = question_transform
        self.answer_transform = answer_transform

        # Check dataset
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # Load dataset
        with open(self.questions_path, encoding='utf-8') as fp:
            questions_json = json.load(fp)

        with open(self.annotations_path, encoding='utf-8') as fp:
            annotations_json = json.load(fp)

        annotations = {}
        for annotation in annotations_json['annotations']:
            annotations[annotation['question_id']] = annotation

        self._questions = []
        for question in questions_json['questions']:
            annotation = annotations[question['question_id']]
            question_text = question['question']
            image_id = question['image_id']
            answer = annotation['multiple_choice_answer']
            if not filter_func(question_text, answer):
                continue
            self._questions.append((image_id, question_text, answer))

        self._coco_zip = zipfile.ZipFile(self.coco_path, mode='r')

        # Create text transform
        if question_transform_factory is not None:
            corpus = [q[1] for q in self._questions]
            self.question_transform = question_transform_factory(corpus)

        if answer_transform_factory is not None:
            corpus = [q[2] for q in self._questions]
            self.answer_transform = answer_transform_factory(corpus)

    @property
    def group(self) -> str:
        return 'train' if self.train else 'val'

    @property
    def coco_path(self) -> str:
        return os.path.join(self.root, f'{self.group}2014.zip')

    @property
    def questions_path(self) -> str:
        filename = f'v2_OpenEnded_mscoco_{self.group}2014_questions.json'
        return os.path.join(self.root, filename)

    @property
    def annotations_path(self) -> str:
        filename = f'v2_mscoco_{self.group}2014_annotations.json'
        return os.path.join(self.root, filename)

    def _check_exists(self) -> bool:
        return os.path.exists(self.coco_path)\
            and os.path.exists(self.questions_path)\
            and os.path.exists(self.annotations_path)

    def download(self) -> None:
        """ Download the VQA2 dataset if it doesn't exist """
        if self._check_exists():
            return
        for resource, extract in self.resources:
            if extract:
                download_and_extract_archive(resource, self.root, remove_finished=True)
            else:
                download_url(resource, self.root)

    def close(self):
        """ Close the COCO dataset zip file """
        self._coco_zip.close()

    def __len__(self):
        return len(self._questions)

    def _get_image(self, image_id: int) -> Image:
        filename = '{0}2014/COCO_{0}2014_{1:012}.jpg'.format(self.group, image_id)
        file_content = io.BytesIO()
        with self._coco_zip.open(filename, 'r') as fp:
            file_content.write(fp.read())
        image = Image.open(file_content)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def __getitem__(self, idx):
        image_id, question, answer = self._questions[idx]
        image = self._get_image(image_id)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.question_transform is not None:
            question = self.question_transform(question)

        if self.answer_transform is not None:
            answer = self.answer_transform(answer)

        return image, question, answer


class EasyVQADataset(Dataset):
    """ EasyVQA Dataset """

    def __init__(
            self,
            train: bool = True,
            image_transform: Optional[Callable] = None,
            text_transform: Optional[Callable] = None,
            text_transform_factory: Optional[Callable[[Iterable[str]], Callable]] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        """
        Args:
            root (string): Root directory of dataset
            train (bool, optional): If True, creates the train set,
                otherwise creates the validation set
            image_transform (callable, optional): Optional transform to be applied on the input PIL image.
            text_transform (callable, optional): Optional transform to be applied on the input question string.
            text_transform_factory (callable, optional): Optional tranform factory to override text_transform;
                the factory function should take in training corpus(list of string) and return a text_transform
                function.
        """
        super().__init__()
        self.train = train
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.target_transform = target_transform

        import easy_vqa

        if self.train:
            data = easy_vqa.get_train_questions()
            paths = easy_vqa.get_train_image_paths()
        else:
            data = easy_vqa.get_test_questions()
            paths = easy_vqa.get_test_image_paths()

        self.questions = []
        self.answers = []
        self.image_paths = []
        for question, answer, image_id in zip(*data):
            if answer != 'yes' and answer != 'no':
                continue
            self.questions.append(question)
            self.answers.append(answer == 'yes')
            self.image_paths.append(paths[image_id])

        if text_transform_factory is not None:
            self.text_transform = text_transform_factory(self.questions)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        question_text = self.questions[idx]
        answer = self.answers[idx]

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.text_transform is not None:
            question_text = self.text_transform(question_text)

        if self.target_transform is not None:
            answer = self.target_transform(answer)

        return image, question_text, answer
