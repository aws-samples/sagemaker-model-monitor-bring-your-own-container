#  Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import os
import pickle
import pathlib

from io import StringIO

import pandas as pd

import sagemaker_xgboost_container.encoder as xgb_encoders


script_path = pathlib.Path(__file__).parent.absolute()
with open(f'{script_path}/preprocess.pkl', 'rb') as f:
    preprocess = pickle.load(f) 


def input_fn(request_body, content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.

    Return a DMatrix (an object that can be passed to predict_fn).
    """

    if content_type == "text/csv":        
        df = pd.read_csv(StringIO(request_body), header=None)
        X = preprocess.transform(df)
        
        X_csv = StringIO()
        pd.DataFrame(X).to_csv(X_csv, header=False, index=False)
        req_transformed = X_csv.getvalue().replace('\n', '')
                
        return xgb_encoders.csv_to_dmatrix(req_transformed)
    else:
        raise ValueError(
            "Content type {} is not supported.".format(content_type)
        )
        

def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    
    model_file = "xgboost-model"
    booster = pickle.load(open(os.path.join(model_dir, model_file), "rb"))
        
    return booster      
