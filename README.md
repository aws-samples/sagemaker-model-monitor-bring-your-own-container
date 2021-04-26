# Bring your own container to project model accuracy drift with Amazon SageMaker Model Monitor

The world we live in is constantly changing and so is the data that is collected to build models. One of the problems that is constantly seen in production environment is that the deployed model is not behaving the same way as it was during the training phase. This concept is generally called as *Data Drift* or *Dataset Shift* and can be caused by many factors such as bias in sampling data that affects features or label data, non-stationary nature of time series data, or changes in data pipeline. Since machine learning models are not deterministic, it is important to minimize the variance in the production environment by periodically monitoring the deployment environment for model drift and sending alerts and if necessary trigger re-training of the models with new data.

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service that enables developers and data scientists to quickly and easily build, train, and deploy ML models at any scale. After you train an ML model, you can deploy it on SageMaker endpoints that are fully managed and can serve inferences in real time with low latency. After you deploy your model, you can use Amazon SageMaker Model Monitor to continuously monitor the quality of your ML model in real time. You can also configure alerts to notify and trigger actions if any drift in model performance is observed. Early and proactive detection of these deviations enables you to take corrective actions, such as collecting new ground truth training data, retraining models, and auditing upstream systems, without having to manually monitor models or build additional tooling.

In this repository, we will present techniques to detect covariate drift, and demonstrate how to incorporate your own custom drift detection algorithms and visualizations with SageMaker model monitor.

## Contents
* `sm_model_monitor.ipynb`: The main SageMaker notebook that will connect all the above data source and scripts.
* `Dockerfile`: The docker file for custom model monitor container.
* `src`: contains files that are used to detect model drift using custom algorithms with SageMaker Model Monitor.  
* `data`: We have chosen [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Adult) from UCI Machine Learning Repository. The dataset consists of people income and several attributes describe demographics of the population. The task is predict if a person makes above or below $50,000. This dataset contains both categorical and integral attributes, and has several missing values. The `data` folder contains training and test datasets, and also data that will be used during inference.
* `model`: contains the XGBoost model trained using `sm_train_xgb.ipynb`script.
* `script`: This folder contains scripts used during model inference.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

