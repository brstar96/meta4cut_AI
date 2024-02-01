import os
import sys
import time
import importlib
import signal
import re
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import errors
from modules.call_queue import queue_lock
# from modules import import_hook, errors, extra_networks, ui_extra_networks_checkpoints
# from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
# from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.paths
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork


def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "1.13.1"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
        You are running torch {torch.__version__}.
        The program is tested to work with torch {expected_torch_version}.
        To reinstall the desired version, run with commandline flag --reinstall-torch.
        Beware that this will cause a lot of large files to be downloaded, as well as
        there are reports of issues with training tab on the latest version.

        Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.16rc425"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
            You are running xformers {xformers.__version__}.
            The program is tested to work with xformers {expected_xformers_version}.
            To reinstall the desired version, run with commandline flag --reinstall-xformers.

            Use --skip-version-check commandline argument to disable this check.
            """.strip())

def initialize():
    check_versions()
    extensions.list_extensions()
    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    codeformer.setup_model(cmd_opts.codeformer_models_path)
    gfpgan.setup_model(cmd_opts.gfpgan_models_path)

    modelloader.list_builtin_upscalers()
    modules.scripts.load_scripts()
    modelloader.load_upscalers()

    modules.sd_vae.refresh_vae_list()

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()

def setup_cors(app):
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])

def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api

# serving with uvicorn
def api_only():
    initialize()

    app = FastAPI()
    setup_cors(app)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)

# testing with local GPU
def local_only():
    initialize()