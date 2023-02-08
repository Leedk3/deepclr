#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import os.path as osp

import torch
from torch import nn

from deepclr.config import load_model_config
from deepclr.data import create_input_dataflow
from deepclr.evaluation import load_scenario, Evaluator
from deepclr.models import load_trained_model, ModelInferenceHelper
from deepclr.utils.logging import create_logger

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import time

import dash
from dash import html
from dash import dcc
import numpy as np

from dash.dependencies import Input, Output, State


SCENE_NUM = 250

app = dash.Dash(__name__, update_title=None)  # remove "Updating..." from title


def main():
    # parse inputs
    parser = argparse.ArgumentParser(description="Model inference for evaluation scenario.")
    parser.add_argument('scenario', type=str, help="scenario configuration (*.yaml)")
    parser.add_argument('model_name', type=str, help="model name (subdirectory of MODEL_PATH)")
    parser.add_argument('output_base', type=str, help="base directory for inference output")
    parser.add_argument('--model_path', type=str, default=None,
                        help="alternative model path instead of MODEL_PATH")
    parser.add_argument('--weights', type=str, default='weights.tar', help="model weights (default: weights.tar)")
    args = parser.parse_args()

    # logging
    logger = create_logger('evaluation')

    # load scenario
    logger.info("Loading scenario")
    scene_cfg = load_scenario(args.scenario, with_method=False)

    # filenames and directories
    model_base_path = args.model_path
    if model_base_path is None:
        model_base_path = os.getenv('MODEL_PATH')
        if model_base_path is None:
            raise RuntimeError("Could not get model path from environment variable MODEL_PATH or argument.")

    model_path = osp.join(model_base_path, args.model_name)
    model_file = osp.join(model_path, 'model_config.yaml')
    weights_file = osp.join(model_path, args.weights)

    # read model config
    logger.info("Read model configuration")
    model_cfg = load_model_config(model_file, weights_file)

    # load model
    logger.info(f"Load model {weights_file}")
    model = load_trained_model(model_cfg)
    model = model.cuda()

    # initialize model inference helper and evaluator
    helper = ModelInferenceHelper(model, is_sequential=scene_cfg.sequential)
    evaluator = Evaluator()

    # create output directory
    output_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = osp.join(args.output_base, f'{output_stamp}_{scene_cfg.name}_{model_cfg.model_type.name}')

    logger.info("Create output directory")
    os.makedirs(output_dir, exist_ok=True)

    # create and store evaluation config
    eval_cfg = scene_cfg.copy()
    eval_cfg.method.name = model_cfg.model_type.name
    eval_cfg.method.params.model_name = args.model_name
    eval_cfg.method.params.model_file = model_file
    eval_cfg.method.params.weights_file = weights_file
    eval_cfg.write_file(osp.join(output_dir, 'scenario.yaml'), invalid=True, internal=True)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    # iterate files
    plotly_vis = True


    df = create_input_dataflow(scene_cfg.dataset_type, scene_cfg.data['08'], shuffle=False)

    # iterate
    df.reset_state()
    helper.reset_state()
    first_call = False

    df = list(df)
    global SCENE_NUM
    ds = df[SCENE_NUM]
    template = torch.from_numpy(ds['clouds'][0]).cuda()
    source = torch.from_numpy(ds['clouds'][1]).cuda()
    stamp = ds['timestamps'][0]
    transform_gt = ds['transform']



    # predict with timing
    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    t_start.record()
    if scene_cfg.sequential:
        if not helper.has_state():
            helper.predict(template)
        y_pred = helper.predict(source)
        # print("partial_predict", y_pred)
        source_partial = helper.partial_predict(source).squeeze(0)
        source_partial = source_partial.detach().cpu().numpy()
        source_partial = source_partial[:3, :] #feature
        source_partial = np.transpose(source_partial) #feature
        # source_partial = source_partial[:, :3] #transformer
        # print(source_partial.shape, source_partial[:, :3]) #transformer

        target_partial = helper.partial_predict(template).squeeze(0)
        target_partial = target_partial.detach().cpu().numpy()
        target_partial = target_partial[:3, :] #feature
        print(target_partial.shape, target_partial) #feature extraction
        target_partial = np.transpose(target_partial) #feature
        print(target_partial.shape, target_partial) #feature extraction
        # target_partial = target_partial[:, :3] #transformer
        # print(target_partial.shape, target_partial[:, :3]) #transformer



    else:
        y_pred = helper.predict(source, template)                

    t_end.record()
    torch.cuda.synchronize()

    source_np = np.array(ds['clouds'][0])
    source_pts_x = source_np[:, 0]
    source_pts_y = source_np[:, 1]
    source_pts_z = source_np[:, 2]
    source_scat = ax.scatter(source_pts_x, source_pts_y,source_pts_z, s=0.01, c='b')

    target_np = np.array(ds['clouds'][1])
    target_pts_x = target_np[:, 0]
    target_pts_y = target_np[:, 1]
    target_pts_z = target_np[:, 2]
    target_scat = ax.scatter(target_pts_x, target_pts_y, target_pts_z,  s=0.01, c='r')
    # print(source_partial)
    source_partial_x = source_partial[:, 0]
    source_partial_y = source_partial[:, 1]
    source_partial_z = source_partial[:, 2]
    attention_scat = ax.scatter(source_partial_x, source_partial_y, source_partial_z,  s=0.5, c='r')

    target_partial_x = target_partial[:, 0]
    target_partial_y = target_partial[:, 1]
    target_partial_z = target_partial[:, 2]

    # plt.show()
    if plotly_vis:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(mode='markers', x=source_pts_x, y=source_pts_y, z=source_pts_z, 
                    marker=dict(
                        color='rgba(255, 0, 25, 0.9)',
                        size=3,
                        # line=dict(
                        #     color='MediumPurple',
                        #     width=1
                        # )
                    )
                    ,name='source'))
        fig.add_trace(go.Scatter3d(mode='markers', x=target_pts_x, y=target_pts_y, z=target_pts_z, 
                    marker=dict(
                        color='rgba(0, 25, 255, 0.9)',
                        size=3,
                        # line=dict(
                        #     color='MediumPurple',
                        #     width=1
                        # )
                    )
                    ,name='target'))
        fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.25)))
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.write_html("plot.html")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter3d(mode='markers', x=source_pts_x, y=source_pts_y, z=source_pts_z, 
                    marker=dict(
                        color='rgba(255, 0, 25, 0.2)',
                        size=3,
                        # line=dict(
                        #     color='MediumPurple',
                        #     width=1
                        # )
                    )
                    ,name='source'))
        fig2.add_trace(go.Scatter3d(mode='markers', x=target_pts_x, y=target_pts_y, z=target_pts_z, 
                    marker=dict(
                        color='rgba(0, 25, 255, 0.2)',
                        size=3,
                        # line=dict(
                        #     color='MediumPurple',
                        #     width=1
                        # )
                    )
                    ,name='target'))
        fig2.add_trace(go.Scatter3d(mode='markers', x=source_partial_x, y=source_partial_y, z=source_partial_z, 
                    marker=dict(
                        color='rgba(255, 0, 25, 0.9)',
                        size=15,
                        # line=dict(
                        #     color='MediumPurple',
                        #     width=2
                        # )
                    )
                    ,name='sampled'))
        # fig2.add_trace(go.Scatter3d(mode='markers', x=target_partial_x, y=target_partial_y, z=target_partial_z, 
        #             marker=dict(
        #                 color='rgba(0, 25, 255, 0.9)',
        #                 size=15,
        #                 # line=dict(
        #                 #     color='MediumPurple',
        #                 #     width=2
        #                 # )
        #             )
        #             ,name='sampled'))


        # Show the fig2ure
        # fig2.update_layout(width=700, margin=dict(r=40, l=40, b=10, t=10))
        # fig2.update_layout(scene=dict(xaxis=dict(autorange='reversed', range=[-60, 60]),
        #                              yaxis=dict(autorange='reversed', range=[-60, 60]),
        #                              zaxis=dict(autorange='reversed', range=[0, 5])))
        fig2.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.25)))
        fig2.update_layout(scene=dict(aspectmode='data'))
        fig2.write_html("plot2.html")
        time.sleep(0.2)

        del df
    
        app.run_server(debug=True)


if __name__ == '__main__':
    main()
