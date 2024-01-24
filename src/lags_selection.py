from copy import deepcopy
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from numpy import abs as nabs, arange, ndarray
from pandas import DataFrame
from shap import (
    Explainer,
    TreeExplainer,
    GPUTreeExplainer,
    LinearExplainer,
    PermutationExplainer,
    SamplingExplainer,
    DeepExplainer,
    KernelExplainer,
    GradientExplainer,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
from typing import Tuple, Callable, Union, List, Literal


from src.subsets import Subsets
from src.plot_tools import plot_scoring, plot_rfe_score, plot_paths


class LagsSelector(BaseEstimator):
    """The LagsSelector class is a wrapper for the MLForecast class that implements a recursive feature elimination based on SHAP values to select the best lags for a given forecaster.
    The method is based on : https://medium.com/towards-data-science/your-features-are-important-it-doesnt-mean-they-are-good-ff468ae2e3d4.
    Note: The "features" of the LagsSelector are the lags of the forecaster.

    Args:
        BaseEstimator (BaseEstimator): The scikit learn base estimator from which the selector inherit.
    """

    def __init__(
        self,
        estimator: RegressorMixin,
        freq: Union[int, str] = 1,
        lags: Union[ndarray, list] = arange(1, 12),
        target_transforms: List[Callable] = [Differences([1])],
    ) -> None:
        self.estimator = estimator
        self.freq = freq
        self.features = lags.tolist() if isinstance(lags, ndarray) else lags
        self.target_transforms = target_transforms

        self.best_forecaster_ = None
        self.best_score_ = None
        self.best_params_ = None

        self.error_rfe_results_ = None
        self.prediction_rfe_results_ = None

    def fit(
        self,
        X: DataFrame,
        explainer: str,
        method: str = Literal["error", "prediction", "both"],
        plot: bool = True,
        save_path: str = None,
        y=None,
    ) -> "LagsSelector":
        """Fit the LagsSelector to the dataset and returns the fitted selector with the best_forecaster and the result from the recursive feature elimination.

        Args:
            X (DataFrame): The dataset to fit the forecaster on.
            explainer (str): The SHAP explainer used to compute the feature contributions. Must be one of ["TreeExplainer", "GPUTreeExplainer", "LinearExplainer", "PermutationExplainer", "SamplingExplainer", "DeepExplainer", "KernelExplainer", "GradientExplainer"].
            method (str, optional): The method to use when performing the recursive feature elimination. If both, both methods will be performed and the one giving the best score on the test set will be keeped. Defaults to Literal["error", "prediction", "both"].
            plot (bool, optional): Whether to plot the RFE results or not. Defaults to True.
            save_path (str, optional): If plot is True, the path to save the plot. Defaults to None.
            y (_type_, optional): For API consistence. Unused. Defaults to None.

        Returns:
            LagsSelector: The fitted selector.
        """
        forecaster = self.__build_forecaster(
            self.estimator, self.features, self.freq, self.target_transforms
        )
        self.sets = Subsets(X, forecaster)
        if method == "both":
            error_fcster, error_score = self.__perform_recursive_feature_elimination(
                forecaster, explainer, "error"
            )
            pred_fcster, pred_score = self.__perform_recursive_feature_elimination(
                forecaster, explainer, "prediction"
            )

            self.best_forecaster_, self.best_score_ = (
                (error_fcster, error_score)
                if error_score < pred_score
                else (pred_fcster, pred_score)
            )
        else:
            self.best_forecaster_, _ = self.__perform_recursive_feature_elimination(
                forecaster, explainer, method
            )

        self.best_forecaster_.fit(self.sets.dataset)

        if plot:
            plot_rfe_score(
                prediction_results=self.prediction_rfe_results_.loc[:, "mae_tst"]
                if self.prediction_rfe_results_ is not None
                else None,
                error_results=self.error_rfe_results_.loc[:, "mae_tst"]
                if self.error_rfe_results_ is not None
                else None,
                save_path=save_path,
            )
        return self

    def predict(self, h: int) -> DataFrame:
        """Used the best forecaster to predict the next h steps.

        Args:
            h (int): The next h steps to predict.

        Raises:
            ValueError: Raise an error if the forecaster is not fitted.

        Returns:
            DataFrame: The prediction dataset.
        """
        if self.best_forecaster_ is None:
            raise ValueError(
                "The forecaster must be fitted before predicting. Please call the fit method first."
            )
        return self.best_forecaster_.predict(h)

    def score(
        self,
        scoring: Callable = mean_absolute_error,
        score_name: str = None,
        plot: bool = True,
        show_train: bool = True,
        path: str = None,
    ) -> float:
        """Compute the score of the best forecaster on the test set.

        Args:
            scoring (Callable, optional): The scoring function to use. Defaults to mean_absolute_error.
            score_name(str, optional): The name of the score to display. Defaults to None.
            plot (bool, optional): Whether to plot the test/preds or not. Defaults to True.
            show_train (bool, optional): If plot is True, whether to plot the train set or not. Defaults to True.
            path (str, optional): If plot is True, the path to save the plot. Defaults to None.

        Raises:
            ValueError: Raise an error if the forecaster is not fitted.

        Returns:
            float: The computed score.
        """
        if self.best_forecaster_ is None:
            raise ValueError(
                "The forecaster must be fitted before predicting. Please call the fit method first."
            )
        fcster_eval = deepcopy(self.best_forecaster_)
        fcster_eval.fit(self.sets.train)
        preds = fcster_eval.predict(self.sets.test.shape[0])
        score = scoring(self.sets.test.loc[:, "y"], preds.iloc[:, -1])
        if plot:
            plot_scoring(
                sets=self.sets,
                preds=preds,
                score=score,
                score_name=scoring.__name__ if score_name is None else score_name,
                show_train=show_train,
                save_path=path,
            )

        return score

    def get_params(self, deep=True) -> dict:
        """Returns the parameters of the selector.

        Args:
            deep (bool, optional): If True, will return the parameters for this estimator and contained subobjects that are estimators. Defaults to True.

        Returns:
            dict: The parameters of the selector.
        """
        return {
            "best_forecaster_": self.best_forecaster_ if deep else None,
            "best_params_": self.best_params_,
            "best_score_": self.best_score_,
            "error_rfe_results_": self.error_rfe_results_,
            "prediction_rfe_results_": self.prediction_rfe_results_,
        }

    def get_paths(
        self, plot: bool = True, save_path: str = None
    ) -> Tuple[DataFrame, DataFrame]:
        """Return and plot the paths taken by each of the recusive feature elimination methods.

        Args:
            plot (bool, optional): Whether to plot the paths or not. Defaults to True.
            save_path (str, optional): The path to where to save the plots. Defaults to None.

        Raises:
            ValueError: An error if the two paths has not been computed yet.

        Returns:
            Tuple[DataFrame, DataFrame]: The paths taken by each method (prediction and error).
        """
        if self.prediction_rfe_results_ is None or self.error_rfe_results_ is None:
            raise ValueError(
                "The paths must be computed before showing them. Please call the fit method first with method='both'."
            )
        if plot:
            plot_paths(self.prediction_rfe_results_, self.error_rfe_results_, save_path)
        return (
            self.prediction_rfe_results_.loc[:, "lag_drop"],
            self.error_rfe_results_.loc[:, "lag_drop"],
        )

    def __perform_recursive_feature_elimination(
        self, forecaster: MLForecast, explainer: str, method: str
    ):
        """Subroutine of the fit method. Performs the recursive feature elimination process.

        Args:
            forecaster (MLForecast): The MLForecast forecaster to use.
            explainer (str): The SHAP explainer name to use.
            method (str): The method to use in the recursive feature elimination process.

        Returns:
            Tuple[MLForecast, float]: The best forecaster and its score on the validation set.
        """
        return self.__recursive_feature_elimination(
            forecaster=forecaster,
            sets=self.sets,
            explainer=explainer,
            method=method,
        )

    def __compute_contributions(
        self, shap_explainer: Explainer, sets: Subsets
    ) -> Tuple[DataFrame, DataFrame]:
        """Compute the contributions of each feature to the prediction and the error.

        Args:
            shap_explainer (Explainer): The SHAP explainer to use to get the feature contributions.
            sets (Subsets): The sets to compute the contributions on.

        Returns:
            Tuple[DataFrame, DataFrame]: The prediction and error contributions.
        """
        shap_values = self.__compute_shap_values(shap_explainer, sets)
        prediction_contribution = self.__compute_prediction_contribution(shap_values)

        error_contribution = self.__compute_error_contribution(shap_values, sets)
        return prediction_contribution, error_contribution

    def __build_forecaster(
        self,
        estimator: RegressorMixin,
        features: list,
        freq: int,
        target_transforms: List[Callable],
    ) -> MLForecast:
        """Builds a MLForecast forecaster with the given parameters.

        Args:
            estimator (RegressorMixin): The ML model to use as estimator.
            features (list): The features to use.
            freq (int): The frequency of the dataset.
            target_transforms (List[Callable]): The target transforms to use.

        Returns:
            MLForecast: The builded forecaster.
        """
        return MLForecast(
            models=[estimator],
            lags=features,
            freq=freq,
            target_transforms=target_transforms,
        )

    def __build_forecaster_and_sets(
        self,
        estimator: RegressorMixin,
        features: list,
        freq: int,
        target_transforms: List[Callable],
        X: DataFrame,
    ) -> Tuple[MLForecast, Subsets]:
        """Build a MLForecast forecaster and subsets from the given parameters.

        Args:
            estimator (RegressorMixin): The estimator to use in the forecaster.
            features (list): The lags to use in the forecaster.
            freq (int): The freq of the dataset.
            target_transforms (List[Callable]): The target transformations to use.
            X (DataFrame): The dataset to build the subsets from (in nixtla format).

        Returns:
            Tuple[MLForecast, Subsets]: The builded forecaster and subsets.
        """
        forecaster = self.__build_forecaster(
            estimator, features, freq, target_transforms
        )
        return forecaster, Subsets(X, forecaster)

    def __get_explainer(
        self, shap_explainer: str, estimator: BaseEstimator, data: DataFrame
    ) -> Explainer:
        """Instantiate the SHAP explainer object to use.

        Args:
            shap_explainer (Explainer): The SHAP explainer object to use. The explainer must include a "shap_values" method.
            estimator (BaseEstimator): The estimator used in the SHAP explainer.
            data (DataFrame): The data used in the SHAP explainer.

        Raises:
            ValueError: Raise an error if the explainer is not known.

        Returns:
            Explainer: The instantiated SHAP explainer.
        """
        match shap_explainer:
            case "TreeExplainer":
                return TreeExplainer(model=estimator, masker=None)
            case "GPUTreeExplainer":
                return GPUTreeExplainer(model=estimator, masker=None)
            case "LinearExplainer":
                return LinearExplainer(model=estimator, masker=data)
            case "PermutationExplainer":
                return PermutationExplainer(model=estimator.predict, masker=data)
            case "SamplingExplainer":
                return SamplingExplainer(model=estimator.predict, data=data)
            case "DeepExplainer":
                return DeepExplainer(model=estimator, data=data)
            case "KernelExplainer":
                return KernelExplainer(model=estimator.predict, data=data)
            case "GradientExplainer":
                return GradientExplainer(model=estimator, data=data)
            case _:
                raise ValueError("Unknown explainer.")

    def __compute_shap_values(
        self, shap_explainer: Explainer, sets: Subsets
    ) -> DataFrame:
        """Compute the SHAP values of the given forecaster on the given sets.

        Args:
            shap_explainer (Explainer): The SHAP explainer to use. Must implement a "shap_values" method.
            sets (Subsets): The sets to compute the SHAP values on.

        Returns:
            DataFrame: The SHAP values.
        """

        self.estimator.fit(sets.X_train, sets.y_train)
        return DataFrame(
            data=shap_explainer.shap_values(sets.X_eval),
            index=sets.X_eval.index,
            columns=sets.X_eval.columns,
        )

    def __compute_prediction_contribution(self, shap_values: DataFrame) -> DataFrame:
        """Compute the prediction contribution of each feature based on the SHAP values.

        Args:
            shap_values (DataFrame): The SHAP values to base the computation on.

        Returns:
            DataFrame: The prediction contribution of each feature.
        """
        return shap_values.abs().mean().rename("prediction_contribution")

    def __compute_error_contribution(
        self, shap_values: DataFrame, sets: Subsets
    ) -> DataFrame:
        """Compute the error contribution of each feature based on the SHAP values.

        Args:
            shap_values (DataFrame): The SHAP values to base the computation on.

        Returns:
            DataFrame: The error contribution of each feature.
        """
        preds = self.estimator.predict(sets.X_eval)
        abs_error = nabs((sets.y_eval - preds))
        y_pred_wo_feature = shap_values.apply(lambda feature: preds - feature)
        abs_error_wo_feature = y_pred_wo_feature.apply(
            lambda feature: (sets.y_eval - feature).abs()
        )
        return (
            abs_error_wo_feature.apply(lambda feature: abs_error - feature)
            .mean()
            .rename("error_contribution")
        )

    def __calculate_ref_mae(self, forecaster: MLForecast, sets: Subsets) -> float:
        """Compute the MAE of the forecaster on the validation set with all lags included.

        Args:
            forecaster (MLForecast): The forecaster to use.
            sets (Subsets): The Subsets object containing the training and validation sets.

        Returns:
            float: The MAE.
        """
        return mean_absolute_error(
            forecaster.fit(sets.train).predict(sets.validation.shape[0]).iloc[:, -1],
            sets.validation.loc[:, "y"],
        )

    def __initialize_rfe(self) -> Tuple[DataFrame, List[str]]:
        """Initialize the recursive feature elimination process.

        Returns:
            Tuple[DataFrame, List[str]]: The dataframe containing the results of the recursive feature elimination process and the list of features.
        """
        self.best_params_ = deepcopy(self.features)
        features = deepcopy(self.features)
        rfe_results = DataFrame(dtype=float)
        return rfe_results, features

    def __calculate_contributions(
        self, shap_explainer: Explainer, new_sets: Subsets
    ) -> Tuple[DataFrame, DataFrame]:
        """Calculate the contributions of each feature to the prediction and the error using the SHAP values.

        Args:
            shap_explainer (Explainer): The SHAP explainer to use to compute the SHAP values.
            new_sets (Subsets): The Subsets object containing the preprocessed training and validation sets.

        Returns:
            Tuple[DataFrame, DataFrame]: The prediction and error contribution.
        """
        return self.__compute_contributions(shap_explainer, new_sets)

    def __fit_and_predict(self, candidate: MLForecast, sets: Subsets) -> float:
        """Fit the candidate forecaster on the training set and predict the validation set.

        Args:
            candidate (MLForecast): The candidate forecaster to fit and predict.
            sets (Subsets): The Subsets object containing the training and validation sets.

        Returns:
            float: The MAE of the candidate forecaster on the validation set.
        """
        candidate.fit(sets.train)
        return mean_absolute_error(
            candidate.predict(sets.validation.shape[0]).iloc[:, -1],
            sets.validation.loc[:, "y"],
        )

    def __update_best_params(
        self, mae_test: float, ref_mae: float, features: List[str]
    ) -> float:
        """Update the best parameters list if the candidate forecaster has a better MAE on the validation set than the reference MAE.

        Args:
            mae_test (float): The MAE of the candidate forecaster on the validation set.
            ref_mae (float): The reference MAE.
            features (List[str]): The lags considered of the candidate forecaster.

        Returns:
            float: The new reference MAE.
        """
        if mae_test < ref_mae:
            self.best_params_ = deepcopy(features)
            ref_mae = mae_test
        return ref_mae

    def __update_rfe_results(
        self,
        iteration: int,
        rfe_results: DataFrame,
        features: list,
        candidate: MLForecast,
        new_sets: Subsets,
        sets: Subsets,
        mae_test: float,
    ) -> None:
        """Update the results of the recursive feature elimination process at each iteration.

        Args:
            iteration (int): The iteration number.
            rfe_results (DataFrame): The dataframe containing the results of the recursive feature elimination process.
            features (list): The features at the iteration.
            candidate (MLForecast): The candidate forecaster.
            new_sets (Subsets): The Subsets object containing the preprocessed training and validation sets.
            sets (Subsets): The Subsets object containing the training and validation sets.
            mae_test (float): The MAE of the candidate forecaster on the validation set.
        """
        candidate_estimator = list(candidate.__dict__["models"].values())[0]
        rfe_results.loc[iteration, "n_features"] = len(features)
        rfe_results.loc[iteration, "mae_trn"] = mean_absolute_error(
            candidate_estimator.predict(new_sets.X_train),
            new_sets.y_train,
        )
        rfe_results.loc[iteration, "mae_tst"] = mae_test
        rfe_results.loc[iteration, "R2_trn"] = r2_score(
            candidate_estimator.predict(new_sets.X_train), new_sets.y_train
        )
        rfe_results.loc[iteration, "R2_tst"] = r2_score(
            candidate.predict(sets.validation.shape[0]).iloc[:, -1],
            sets.validation.loc[:, "y"],
        )

    def __drop_feature(
        self,
        iteration: int,
        rfe_results: DataFrame,
        features: List[str],
        method: Literal["error", "prediction"],
        prediction_contribution: DataFrame,
        error_contribution: DataFrame,
    ) -> List[str]:
        """Drop the feature with the lowest contribution to the prediction or the highest contribution on the error.

        Args:
            iteration (int): The iteration of the recursive feature elimination process.
            rfe_results (DataFrame): The results of the recursive feature elimination process.
            features (List[str]): The lags considered.
            method (Literal["error", "prediction"]): The method to use in the recursive feature elimination process.
            prediction_contribution (DataFrame): The contribution of each feature to the prediction.
            error_contribution (DataFrame): The contribution of each feature to the error.

        Returns:
            List[str]: The new features list.
        """
        if method == "error":
            features.pop(error_contribution.argmax())
            rfe_results.loc[iteration, "lag_drop"] = error_contribution.idxmax()
            rfe_results.loc[iteration, "contrib"] = error_contribution.max()
        else:
            features.pop(prediction_contribution.argmin())
            rfe_results.loc[iteration, "lag_drop"] = prediction_contribution.idxmin()
            rfe_results.loc[iteration, "contrib"] = prediction_contribution.min()
        return features

    def __finalize_rfe(
        self, rfe_results: DataFrame, method: str
    ) -> Tuple[MLForecast, float]:
        best_forecaster_ = MLForecast(
            models=[self.estimator],
            freq=self.freq,
            lags=self.best_params_,
            target_transforms=self.target_transforms,
        )
        rfe_results["lag_drop"] = rfe_results["lag_drop"].apply(
            lambda x: int(x.split("lag")[1])
        )
        if method == "error":
            self.error_rfe_results_ = rfe_results
        else:
            self.prediction_rfe_results_ = rfe_results
        return best_forecaster_, self.best_params_

    def __recursive_feature_elimination(
        self,
        forecaster: MLForecast,
        sets: Subsets,
        explainer: str,
        method: Literal["error", "prediction", "both"],
    ) -> Tuple[MLForecast, float]:
        """Perform the recursive feature elimination process.

        Args:
            forecaster (MLForecast): The forecaster to use.
            sets (Subsets): The Subsets object based on the time serie dataframe.
            shap_explainer (str): The SHAP explainer to use.
            method (Literal["error", "prediction", "both"]): The method to use in the recursive feature elimination process. If both, performs both methods and keep the one giving the best score on the test set.

        Returns:
            Tuple[MLForecast, float]: The best forecaster and its score on the validation set.
        """
        ref_mae = self.__calculate_ref_mae(forecaster, sets)
        rfe_results, features = self.__initialize_rfe()
        for iteration in tqdm(range(len(features))):
            candidate, new_sets = self.__build_forecaster_and_sets(
                estimator=self.estimator,
                features=features,
                freq=self.freq,
                target_transforms=self.target_transforms,
                X=sets.dataset,
            )
            shap_explainer = self.__get_explainer(
                explainer,
                self.estimator.fit(new_sets.X_train, new_sets.y_train),
                new_sets.X_eval,
            )
            (
                prediction_contribution,
                error_contribution,
            ) = self.__calculate_contributions(shap_explainer, new_sets)
            mae_test = self.__fit_and_predict(candidate, sets)
            ref_mae = self.__update_best_params(mae_test, ref_mae, features)
            self.__update_rfe_results(
                iteration, rfe_results, features, candidate, new_sets, sets, mae_test
            )
            features = self.__drop_feature(
                iteration,
                rfe_results,
                features,
                method,
                prediction_contribution,
                error_contribution,
            )

        best_forecaster_, best_score_ = self.__finalize_rfe(rfe_results, method)
        return best_forecaster_, best_score_
