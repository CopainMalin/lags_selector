from typing import Union, Tuple
from pandas import DataFrame
from dataclasses import dataclass, field
from mlforecast import MLForecast


@dataclass
class Subsets:
    """
    A class that convert a dataset into subsets for training, testing, and validation.

    Parameters:
    - dataset (DataFrame): The original dataset.
    - forecaster (MLForecast): The forecaster object used for preprocessing.
    - test_size (Union[float, int]): The size of the test set as a fraction or an integer.
    - validation_split (Union[float, int]): The fraction or integer to split the training set into validation set.
    """

    dataset: DataFrame
    forecaster: MLForecast
    test_size: Union[float, int] = 0.2
    validation_split: Union[float, int] = 0.2
    train: DataFrame = field(init=False)
    test: DataFrame = field(init=False)
    subtrain: DataFrame = field(init=False)
    validation: DataFrame = field(init=False)
    X_train: DataFrame = field(init=False)
    y_train: DataFrame = field(init=False)
    X_eval: DataFrame = field(init=False)
    y_eval: DataFrame = field(init=False)

    def __post_init__(self) -> None:
        """
        Initializes the class and splits the dataset into train, test, and validation sets.
        Also preprocesses the subtrain set and extracts X_train, y_train, X_eval, and y_eval.

        Returns:
        None
        """
        self.train, self.test = self.__split_dataset(self.dataset, self.test_size)
        self.subtrain, self.validation = self.__split_dataset(
            self.train, self.validation_split
        )
        prep = self.forecaster.preprocess(self.subtrain)
        train_set, eval_set = self.__split_dataset(prep, 0.33)
        self.X_train, self.y_train = self.__extract_Xy(train_set)
        self.X_eval, self.y_eval = self.__extract_Xy(eval_set)

    def __split_dataset(
        self, dataset: DataFrame, test_size: Union[int, float]
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Splits the dataset into a train and a test set based on the test size.

        Parameters:
        - dataset (DataFrame): The dataset to be split.
        - test_size (Union[int, float]): The size of the test set as a fraction or an integer.

        Returns:
        Tuple[DataFrame, DataFrame]: The train and test subsets.
        """
        if isinstance(test_size, int) and test_size < dataset.shape[0]:
            split_index = test_size
        elif isinstance(test_size, float) and test_size < 1:
            split_index = int(test_size * dataset.shape[0])
        else:
            raise ValueError(
                "'test_size' must be either an integer lower than the df size or a float lower than one."
            )
        return dataset.iloc[:-split_index, :], dataset.iloc[-split_index:, :]

    def __extract_Xy(self, preprocessed: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Extracts the feature matrix (X) and the target variable (y) from the preprocessed dataset.

        Parameters:
        - preprocessed (DataFrame): The preprocessed dataset.

        Returns:
        Tuple[DataFrame, DataFrame]: The feature matrix (X) and the target variable (y).
        """
        return (
            preprocessed.drop(["ds", "y", "unique_id"], axis=1),
            preprocessed.loc[:, "y"],
        )
